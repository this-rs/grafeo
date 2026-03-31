//! Benchmark harness for obrain-chat — 12-question evaluation suite.
//!
//! Defines the standard questions, expected keywords, and result structures.
//! Used by `--benchmark` mode in main.rs to evaluate retrieval quality
//! across different KV cache configurations (f16, q8_0, q4_0, etc.).

use serde::{Deserialize, Serialize};

/// Category of a benchmark question.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QuestionCategory {
    DirectRecall,
    MultiHop,
    Relationship,
    Aggregation,
    Context,
}

impl std::fmt::Display for QuestionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirectRecall => write!(f, "direct_recall"),
            Self::MultiHop => write!(f, "multi_hop"),
            Self::Relationship => write!(f, "relationship"),
            Self::Aggregation => write!(f, "aggregation"),
            Self::Context => write!(f, "context"),
        }
    }
}

/// A benchmark question with expected keywords for hit detection.
#[derive(Debug, Clone)]
pub struct BenchQuestion {
    pub query: &'static str,
    pub keywords: &'static [&'static str],
    pub category: QuestionCategory,
}

/// The standard 12-question benchmark suite (from Phase E).
pub const BENCH_QUESTIONS: &[BenchQuestion] = &[
    BenchQuestion {
        query: "Où habite Thomas Rivière ?",
        keywords: &["Lyon"],
        category: QuestionCategory::DirectRecall,
    },
    BenchQuestion {
        query: "Sur quel projet travaille Sophie Martin ?",
        keywords: &["DataPipeline"],
        category: QuestionCategory::DirectRecall,
    },
    BenchQuestion {
        query: "Quel langage utilise Marc Dupont ?",
        keywords: &["Go"],
        category: QuestionCategory::DirectRecall,
    },
    BenchQuestion {
        query: "Qui habite dans la même ville que Thomas Rivière ?",
        keywords: &["Marc"],
        category: QuestionCategory::MultiHop,
    },
    BenchQuestion {
        query: "Quelles technologies sont utilisées par le projet Obrain ?",
        keywords: &["Rust"],
        category: QuestionCategory::MultiHop,
    },
    BenchQuestion {
        query: "Qui connaît Thomas Rivière ?",
        keywords: &["Marc", "Alice"],
        category: QuestionCategory::Relationship,
    },
    BenchQuestion {
        query: "À quel événement Thomas et Marc ont-ils assisté ensemble ?",
        keywords: &["RustConf"],
        category: QuestionCategory::Relationship,
    },
    BenchQuestion {
        query: "Combien de personnes utilisent Python ?",
        keywords: &["Sophie", "Alice"],
        category: QuestionCategory::Aggregation,
    },
    BenchQuestion {
        query: "Liste toutes les villes mentionnées dans le graphe.",
        keywords: &["Lyon", "Paris", "Grenoble", "Toulouse"],
        category: QuestionCategory::Aggregation,
    },
    BenchQuestion {
        query: "Décris le profil professionnel de Thomas Rivière.",
        keywords: &["Rust", "Mozilla", "Obrain"],
        category: QuestionCategory::Context,
    },
    BenchQuestion {
        query: "Quelles sont les compétences de Pierre Bernard ?",
        keywords: &["DevOps", "Bash", "Airbus"],
        category: QuestionCategory::Context,
    },
    BenchQuestion {
        query: "Quel est le lien entre Alice Chen et NeuralSearch ?",
        keywords: &["travaille", "INRIA"],
        category: QuestionCategory::Relationship,
    },
];

/// Result for a single benchmark question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub query: String,
    pub category: QuestionCategory,
    pub hit: bool,
    pub keywords_found: Vec<String>,
    pub keywords_missing: Vec<String>,
    pub latency_ms: f64,
    pub response_preview: String,
    /// Full response text (for debugging; preview is truncated to 120 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_response: Option<String>,
}

/// Summary statistics for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchSummary {
    pub hit_rate: f64,
    pub hits: usize,
    pub total: usize,
    pub latency_avg_ms: f64,
    pub latency_p95_ms: f64,
    pub by_category: std::collections::HashMap<String, CategoryStats>,
}

/// Per-category statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub hit_rate: f64,
    pub hits: usize,
    pub total: usize,
    pub latency_avg_ms: f64,
}

/// Full benchmark report (serialized to JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub model: String,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub n_ctx: u32,
    pub questions: Vec<BenchResult>,
    pub summary: BenchSummary,
}

/// Check if a response contains the expected keywords (case-insensitive).
/// A question is a "hit" if ALL keywords are found in the response.
pub fn check_keywords(response: &str, keywords: &[&str]) -> (bool, Vec<String>, Vec<String>) {
    let response_lower = response.to_lowercase();
    let mut found = Vec::new();
    let mut missing = Vec::new();

    for kw in keywords {
        if response_lower.contains(&kw.to_lowercase()) {
            found.push(kw.to_string());
        } else {
            missing.push(kw.to_string());
        }
    }

    let hit = missing.is_empty();
    (hit, found, missing)
}

/// Compute summary statistics from individual results.
pub fn compute_summary(results: &[BenchResult]) -> BenchSummary {
    let total = results.len();
    let hits = results.iter().filter(|r| r.hit).count();
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    // Latency stats
    let mut latencies: Vec<f64> = results.iter().map(|r| r.latency_ms).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let latency_avg_ms = if total > 0 {
        latencies.iter().sum::<f64>() / total as f64
    } else {
        0.0
    };

    let latency_p95_ms = if total > 0 {
        let idx = ((total as f64 * 0.95).ceil() as usize).min(total) - 1;
        latencies[idx]
    } else {
        0.0
    };

    // Per-category breakdown
    let mut by_category: std::collections::HashMap<String, Vec<&BenchResult>> =
        std::collections::HashMap::new();
    for r in results {
        by_category
            .entry(r.category.to_string())
            .or_default()
            .push(r);
    }

    let cat_stats: std::collections::HashMap<String, CategoryStats> = by_category
        .into_iter()
        .map(|(cat, rs)| {
            let cat_total = rs.len();
            let cat_hits = rs.iter().filter(|r| r.hit).count();
            let cat_latency = rs.iter().map(|r| r.latency_ms).sum::<f64>() / cat_total as f64;
            (
                cat,
                CategoryStats {
                    hit_rate: cat_hits as f64 / cat_total as f64,
                    hits: cat_hits,
                    total: cat_total,
                    latency_avg_ms: cat_latency,
                },
            )
        })
        .collect();

    BenchSummary {
        hit_rate,
        hits,
        total,
        latency_avg_ms,
        latency_p95_ms,
        by_category: cat_stats,
    }
}

/// Format KV type integer to human-readable name.
pub fn kv_type_name(t: i32) -> &'static str {
    match t {
        1 => "f16",
        2 => "q4_0",
        6 => "q5_0",
        8 => "q8_0",
        10 => "q3_0",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_keywords_all_found() {
        let (hit, found, missing) = check_keywords("Thomas habite à Lyon depuis 2020", &["Lyon"]);
        assert!(hit);
        assert_eq!(found, vec!["Lyon"]);
        assert!(missing.is_empty());
    }

    #[test]
    fn test_check_keywords_case_insensitive() {
        let (hit, found, _) = check_keywords("Il utilise RUST et python", &["Rust", "Python"]);
        assert!(hit);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_check_keywords_partial_miss() {
        let (hit, _, missing) =
            check_keywords("Marc utilise Go", &["Go", "Python"]);
        assert!(!hit);
        assert_eq!(missing, vec!["Python"]);
    }

    #[test]
    fn test_compute_summary() {
        let results = vec![
            BenchResult {
                query: "Q1".into(),
                category: QuestionCategory::DirectRecall,
                hit: true,
                keywords_found: vec!["Lyon".into()],
                keywords_missing: vec![],
                latency_ms: 100.0,
                response_preview: "...".into(),
            },
            BenchResult {
                query: "Q2".into(),
                category: QuestionCategory::DirectRecall,
                hit: false,
                keywords_found: vec![],
                keywords_missing: vec!["Go".into()],
                latency_ms: 200.0,
                response_preview: "...".into(),
            },
            BenchResult {
                query: "Q3".into(),
                category: QuestionCategory::MultiHop,
                hit: true,
                keywords_found: vec!["Marc".into()],
                keywords_missing: vec![],
                latency_ms: 150.0,
                response_preview: "...".into(),
            },
        ];

        let summary = compute_summary(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.hits, 2);
        assert!((summary.hit_rate - 0.6667).abs() < 0.01);
        assert!((summary.latency_avg_ms - 150.0).abs() < 0.01);
        assert_eq!(summary.by_category.len(), 2);
        assert_eq!(summary.by_category["direct_recall"].hits, 1);
        assert_eq!(summary.by_category["multi_hop"].hits, 1);
    }

    #[test]
    fn test_bench_questions_count() {
        assert_eq!(BENCH_QUESTIONS.len(), 12);
    }

    #[test]
    fn test_kv_type_name() {
        assert_eq!(kv_type_name(1), "f16");
        assert_eq!(kv_type_name(8), "q8_0");
        assert_eq!(kv_type_name(2), "q4_0");
    }

    // ─────────────────────────────────────────────────────────────────
    // REGRESSION tests for documented bugs
    // ─────────────────────────────────────────────────────────────────

    /// REGRESSION: check_keywords must be case-insensitive for ALL keywords.
    /// Without this, benchmark scores are wrong because the LLM may produce
    /// "lyon" instead of "Lyon", "rust" instead of "Rust", etc.
    #[test]
    fn test_regression_keywords_case_insensitive_all_variants() {
        // Uppercase in expected, lowercase in response
        let (hit, _, _) = check_keywords("thomas habite à lyon", &["Lyon"]);
        assert!(hit, "Lyon vs lyon must match");

        // Mixed case
        let (hit, _, _) = check_keywords("Il utilise RUST", &["Rust"]);
        assert!(hit, "Rust vs RUST must match");

        // Keyword in middle of word — this IS a match (contains)
        let (hit, _, _) = check_keywords("DataPipeline est un projet", &["DataPipeline"]);
        assert!(hit, "DataPipeline exact match");

        // Multi-keyword: all must match
        let (hit, found, missing) = check_keywords("Marc et Alice connaissent Thomas", &["Marc", "Alice"]);
        assert!(hit, "both keywords found");
        assert_eq!(found.len(), 2);
        assert!(missing.is_empty());
    }

    /// REGRESSION: check_keywords must handle French accented characters.
    /// The benchmark questions use French ("Où", "développeur", "Rivière").
    #[test]
    fn test_regression_keywords_french_accents() {
        // Accented chars in response
        let (hit, _, _) = check_keywords("développeur Rust chez Mozilla", &["Rust", "Mozilla"]);
        assert!(hit);

        // Accent in keyword
        let (hit, _, _) = check_keywords("Thomas Rivière habite Lyon", &["Rivière"]);
        assert!(hit, "Accented keyword must match accented response");
    }

    /// REGRESSION: check_keywords with empty response must return all missing.
    #[test]
    fn test_regression_keywords_empty_response() {
        let (hit, found, missing) = check_keywords("", &["Lyon", "Rust"]);
        assert!(!hit);
        assert!(found.is_empty());
        assert_eq!(missing.len(), 2);
    }

    /// REGRESSION: compute_summary with 0 results must not panic (div by zero).
    #[test]
    fn test_regression_summary_empty_results() {
        let summary = compute_summary(&[]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.hits, 0);
        assert_eq!(summary.hit_rate, 0.0);
        assert_eq!(summary.latency_avg_ms, 0.0);
        assert_eq!(summary.latency_p95_ms, 0.0);
        assert!(summary.by_category.is_empty());
    }

    /// REGRESSION: All 12 benchmark questions must have non-empty keywords.
    /// A question with empty keywords would always "pass" — false positive.
    #[test]
    fn test_regression_all_questions_have_keywords() {
        for (i, q) in BENCH_QUESTIONS.iter().enumerate() {
            assert!(
                !q.keywords.is_empty(),
                "Q{}: must have at least one keyword, otherwise it always passes",
                i + 1
            );
        }
    }

    /// REGRESSION: Keywords must not contain substrings that trivially match stop words.
    /// E.g., if a keyword is "a" it would match almost any French text.
    #[test]
    fn test_regression_keywords_not_too_short() {
        for (i, q) in BENCH_QUESTIONS.iter().enumerate() {
            for kw in q.keywords {
                assert!(
                    kw.len() >= 2,
                    "Q{}: keyword '{}' is too short (< 2 chars), would produce false positives",
                    i + 1,
                    kw
                );
            }
        }
    }
}
