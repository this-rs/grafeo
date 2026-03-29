//! Lightweight Okapi BM25 index for :Message nodes in PersonaDB.
//!
//! No external dependencies — pure Rust, ~200 lines.
//! Indexes message content for full-text search across all conversations.

use obrain_common::types::NodeId;
use std::collections::HashMap;

// ── BM25 parameters (Okapi defaults) ────────────────────────────────────────
const K1: f64 = 1.2;
const B: f64 = 0.75;

// ── Stop words (FR + EN, lightweight set) ───────────────────────────────────
const STOP_WORDS: &[&str] = &[
    // French
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "est", "que", "qui", "dans",
    "pour", "pas", "sur", "ce", "il", "elle", "au", "aux", "son", "sa", "ses", "par", "ne", "se",
    "ou", "mais", "avec", "plus", "tout", "bien", "aussi", "comme", "ont", "mon", "ton", "nous",
    "vous", "ils", "elles", "leur", "cette", "ces", "été", "être", "avoir", "fait", "faire", "dit",
    "car", "dont", "très", "peut", "alors", "quand", "ça", "été", // English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "or", "but", "if", "not", "no", "so",
    "up", "out", "about", "into", "than", "then", "them", "they", "this", "that", "it", "its",
    "and", "he", "she", "we", "you", "my", "your", "his", "her", "our", "all", "what", "which",
    "who", "when", "how", "there", "where",
];

/// A single indexed document.
struct Doc {
    node_id: NodeId,
    /// Conversation NodeId this message belongs to.
    conv_id: NodeId,
    role: String,
    /// Term frequencies: term → count.
    tf: HashMap<String, u32>,
    /// Total number of terms in the document.
    term_count: u32,
}

/// Result of a BM25 search.
#[derive(Debug, Clone)]
pub struct MessageHit {
    pub node_id: NodeId,
    pub conv_id: NodeId,
    pub role: String,
    pub content: String,
    pub score: f64,
}

/// Lightweight BM25 index over :Message nodes.
pub struct MessageIndex {
    docs: Vec<Doc>,
    /// Document frequency: term → number of docs containing it.
    df: HashMap<String, u32>,
    /// Average document length (in terms).
    avg_dl: f64,
    /// Original content stored for retrieval.
    contents: HashMap<NodeId, String>,
    /// Stop words set (built once).
    stop_words: std::collections::HashSet<&'static str>,
}

impl Default for MessageIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageIndex {
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            df: HashMap::new(),
            avg_dl: 0.0,
            contents: HashMap::new(),
            stop_words: STOP_WORDS.iter().copied().collect(),
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Add a message to the index.
    pub fn add(&mut self, node_id: NodeId, conv_id: NodeId, role: &str, content: &str) {
        // Don't re-index if already present
        if self.contents.contains_key(&node_id) {
            return;
        }

        let terms = self.tokenize(content);
        let term_count = terms.len() as u32;

        // Build term frequency map
        let mut tf: HashMap<String, u32> = HashMap::new();
        let mut unique_terms: std::collections::HashSet<String> = std::collections::HashSet::new();
        for term in &terms {
            *tf.entry(term.clone()).or_insert(0) += 1;
            unique_terms.insert(term.clone());
        }

        // Update document frequency
        for term in &unique_terms {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }

        // Store content for retrieval
        self.contents.insert(node_id, content.to_string());

        self.docs.push(Doc {
            node_id,
            conv_id,
            role: role.to_string(),
            tf,
            term_count,
        });

        // Recompute average document length
        self.recompute_avg_dl();
    }

    /// Search the index with BM25 scoring. Returns top `limit` hits.
    pub fn search(&self, query: &str, limit: usize) -> Vec<MessageHit> {
        if self.docs.is_empty() {
            return Vec::new();
        }

        let query_terms = self.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.docs.len() as f64;
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for (idx, doc) in self.docs.iter().enumerate() {
            let mut score = 0.0;
            let dl = doc.term_count as f64;

            for term in &query_terms {
                let tf = *doc.tf.get(term).unwrap_or(&0) as f64;
                if tf == 0.0 {
                    continue;
                }
                let df = *self.df.get(term).unwrap_or(&0) as f64;
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                // TF normalization: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / self.avg_dl));
                score += idf * tf_norm;
            }

            if score > 0.0 {
                scores.push((idx, score));
            }
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);

        scores
            .into_iter()
            .map(|(idx, score)| {
                let doc = &self.docs[idx];
                MessageHit {
                    node_id: doc.node_id,
                    conv_id: doc.conv_id,
                    role: doc.role.clone(),
                    content: self.contents.get(&doc.node_id).cloned().unwrap_or_default(),
                    score,
                }
            })
            .collect()
    }

    /// Tokenize text into lowercase terms, filtering stop words and short terms.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != '\'')
            .filter(|s| {
                // Keep tokens ≥ 3 bytes, OR shorter tokens containing non-ASCII
                // (Greek letters like ζ, α, Ψ are important in math/science contexts)
                s.len() > 2 || !s.is_ascii()
            })
            .filter(|s| !self.stop_words.contains(s))
            .map(|s| s.to_string())
            .collect()
    }

    fn recompute_avg_dl(&mut self) {
        if self.docs.is_empty() {
            self.avg_dl = 0.0;
        } else {
            let total: u64 = self.docs.iter().map(|d| d.term_count as u64).sum();
            self.avg_dl = total as f64 / self.docs.len() as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u64) -> NodeId {
        NodeId(n)
    }

    #[test]
    fn test_basic_search() {
        let mut idx = MessageIndex::new();
        idx.add(
            nid(1),
            nid(100),
            "user",
            "J'habite à Paris et j'aime le Rust",
        );
        idx.add(nid(2), nid(100), "assistant", "Paris est une belle ville");
        idx.add(nid(3), nid(100), "user", "Mon chat s'appelle Minou");
        idx.add(
            nid(4),
            nid(101),
            "user",
            "Je programme en Rust depuis 3 ans",
        );

        let results = idx.search("Rust", 10);
        assert_eq!(results.len(), 2);
        // Both docs mentioning "Rust" should be found
        let ids: Vec<NodeId> = results.iter().map(|h| h.node_id).collect();
        assert!(ids.contains(&nid(1)));
        assert!(ids.contains(&nid(4)));
        // "Mon chat s'appelle Minou" should NOT match
        assert!(!ids.contains(&nid(3)));
    }

    #[test]
    fn test_cross_conversation() {
        let mut idx = MessageIndex::new();
        idx.add(
            nid(1),
            nid(100),
            "user",
            "Mon projet Obrain utilise llama.cpp",
        );
        idx.add(
            nid(2),
            nid(200),
            "user",
            "Comment compiler llama.cpp sur macOS",
        );

        let results = idx.search("llama.cpp", 10);
        assert_eq!(results.len(), 2);
        // Both conversations should be found
        let conv_ids: Vec<NodeId> = results.iter().map(|h| h.conv_id).collect();
        assert!(conv_ids.contains(&nid(100)));
        assert!(conv_ids.contains(&nid(200)));
    }

    #[test]
    fn test_no_duplicate_indexing() {
        let mut idx = MessageIndex::new();
        idx.add(nid(1), nid(100), "user", "Hello world");
        idx.add(nid(1), nid(100), "user", "Hello world"); // duplicate
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_empty_query() {
        let mut idx = MessageIndex::new();
        idx.add(nid(1), nid(100), "user", "test content");
        assert!(idx.search("", 10).is_empty());
        assert!(idx.search("le la", 10).is_empty()); // all stop words
    }

    #[test]
    fn test_bm25_scoring_prefers_specific_terms() {
        let mut idx = MessageIndex::new();
        // Add many docs mentioning "code"
        for i in 0..20 {
            idx.add(
                nid(i),
                nid(100),
                "user",
                &format!("I write code every day #{i}"),
            );
        }
        // Add one doc mentioning "quantum"
        idx.add(
            nid(100),
            nid(100),
            "user",
            "Quantum computing is fascinating code",
        );

        let results = idx.search("quantum code", 5);
        // The doc with "quantum" should rank first (rare term = high IDF)
        assert_eq!(results[0].node_id, nid(100));
    }
}
