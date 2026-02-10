//! Shared keyword definitions for graph query language parsers.
//!
//! Keywords are categorized as:
//! - **Common**: shared across GQL, Cypher, and SQL/PGQ
//! - **Language-specific**: unique to a single parser
//!
//! Each lexer calls `CommonKeyword::from_str` for shared keyword
//! recognition, then maps the result to its own `TokenKind`.
//! Language-specific keywords remain in each lexer.

/// A keyword shared across multiple query language parsers.
///
/// Each parser maps these to its own `TokenKind` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommonKeyword {
    // Query structure
    Match,
    Return,
    Where,
    As,
    Distinct,
    With,
    Optional,

    // Ordering & pagination
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Skip,

    // Logical operators
    And,
    Or,
    Not,

    // Comparison
    In,
    Is,
    Like,

    // String predicates
    Starts,
    Ends,
    Contains,

    // Literals
    Null,
    True,
    False,

    // Mutation
    Create,
    Delete,
    Set,
    Remove,
    Merge,
    Detach,
    On,

    // Subquery / procedure
    Call,
    Yield,
    Exists,
    Unwind,

    // Graph structure
    Node,
    Edge,

    // Aggregate / grouping
    Having,
    Case,
    When,
    Then,
    Else,
    End,
}

impl CommonKeyword {
    /// Recognizes a keyword from its uppercase text.
    ///
    /// Returns `None` for language-specific or unrecognized identifiers.
    /// The caller should convert the input to uppercase before calling.
    #[must_use]
    pub fn from_uppercase(text: &str) -> Option<Self> {
        match text {
            // Query structure
            "MATCH" => Some(Self::Match),
            "RETURN" => Some(Self::Return),
            "WHERE" => Some(Self::Where),
            "AS" => Some(Self::As),
            "DISTINCT" => Some(Self::Distinct),
            "WITH" => Some(Self::With),
            "OPTIONAL" => Some(Self::Optional),

            // Ordering & pagination
            "ORDER" => Some(Self::Order),
            "BY" => Some(Self::By),
            "ASC" => Some(Self::Asc),
            "DESC" => Some(Self::Desc),
            "LIMIT" => Some(Self::Limit),
            "SKIP" => Some(Self::Skip),

            // Logical
            "AND" => Some(Self::And),
            "OR" => Some(Self::Or),
            "NOT" => Some(Self::Not),

            // Comparison
            "IN" => Some(Self::In),
            "IS" => Some(Self::Is),
            "LIKE" => Some(Self::Like),

            // String predicates
            "STARTS" => Some(Self::Starts),
            "ENDS" => Some(Self::Ends),
            "CONTAINS" => Some(Self::Contains),

            // Literals
            "NULL" => Some(Self::Null),
            "TRUE" => Some(Self::True),
            "FALSE" => Some(Self::False),

            // Mutation
            "CREATE" => Some(Self::Create),
            "DELETE" => Some(Self::Delete),
            "SET" => Some(Self::Set),
            "REMOVE" => Some(Self::Remove),
            "MERGE" => Some(Self::Merge),
            "DETACH" => Some(Self::Detach),
            "ON" => Some(Self::On),

            // Subquery / procedure
            "CALL" => Some(Self::Call),
            "YIELD" => Some(Self::Yield),
            "EXISTS" => Some(Self::Exists),
            "UNWIND" => Some(Self::Unwind),

            // Graph structure
            "NODE" => Some(Self::Node),
            "EDGE" => Some(Self::Edge),

            // Aggregate / grouping
            "HAVING" => Some(Self::Having),
            "CASE" => Some(Self::Case),
            "WHEN" => Some(Self::When),
            "THEN" => Some(Self::Then),
            "ELSE" => Some(Self::Else),
            "END" => Some(Self::End),

            _ => None,
        }
    }

    /// Returns true if the given uppercase text is a keyword in any parser.
    #[must_use]
    pub fn is_keyword(text: &str) -> bool {
        Self::from_uppercase(text).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_keywords_recognized() {
        // Query structure
        assert_eq!(
            CommonKeyword::from_uppercase("MATCH"),
            Some(CommonKeyword::Match)
        );
        assert_eq!(
            CommonKeyword::from_uppercase("RETURN"),
            Some(CommonKeyword::Return)
        );
        assert_eq!(
            CommonKeyword::from_uppercase("WHERE"),
            Some(CommonKeyword::Where)
        );

        // Logical
        assert_eq!(
            CommonKeyword::from_uppercase("AND"),
            Some(CommonKeyword::And)
        );
        assert_eq!(CommonKeyword::from_uppercase("OR"), Some(CommonKeyword::Or));
        assert_eq!(
            CommonKeyword::from_uppercase("NOT"),
            Some(CommonKeyword::Not)
        );

        // Literals
        assert_eq!(
            CommonKeyword::from_uppercase("NULL"),
            Some(CommonKeyword::Null)
        );
        assert_eq!(
            CommonKeyword::from_uppercase("TRUE"),
            Some(CommonKeyword::True)
        );
        assert_eq!(
            CommonKeyword::from_uppercase("FALSE"),
            Some(CommonKeyword::False)
        );
    }

    #[test]
    fn test_non_keywords_return_none() {
        assert_eq!(CommonKeyword::from_uppercase("FOOBAR"), None);
        assert_eq!(CommonKeyword::from_uppercase("person"), None);
        assert_eq!(CommonKeyword::from_uppercase(""), None);
    }

    #[test]
    fn test_is_keyword() {
        assert!(CommonKeyword::is_keyword("MATCH"));
        assert!(CommonKeyword::is_keyword("WHERE"));
        assert!(!CommonKeyword::is_keyword("FOOBAR"));
    }

    #[test]
    fn test_language_specific_not_common() {
        // SQL/PGQ specific
        assert_eq!(CommonKeyword::from_uppercase("SELECT"), None);
        assert_eq!(CommonKeyword::from_uppercase("FROM"), None);
        assert_eq!(CommonKeyword::from_uppercase("GRAPH_TABLE"), None);

        // Cypher specific
        assert_eq!(CommonKeyword::from_uppercase("UNION"), None);
        assert_eq!(CommonKeyword::from_uppercase("XOR"), None);

        // GQL specific
        assert_eq!(CommonKeyword::from_uppercase("VECTOR"), None);
        assert_eq!(CommonKeyword::from_uppercase("INDEX"), None);
    }

    #[test]
    fn test_all_common_keywords_covered() {
        let all_keywords = [
            "MATCH", "RETURN", "WHERE", "AS", "DISTINCT", "WITH", "OPTIONAL", "ORDER", "BY", "ASC",
            "DESC", "LIMIT", "SKIP", "AND", "OR", "NOT", "IN", "IS", "LIKE", "STARTS", "ENDS",
            "CONTAINS", "NULL", "TRUE", "FALSE", "CREATE", "DELETE", "SET", "REMOVE", "MERGE",
            "DETACH", "ON", "CALL", "YIELD", "EXISTS", "UNWIND", "NODE", "EDGE", "HAVING", "CASE",
            "WHEN", "THEN", "ELSE", "END",
        ];

        for kw in &all_keywords {
            assert!(
                CommonKeyword::from_uppercase(kw).is_some(),
                "Expected '{kw}' to be recognized as a common keyword"
            );
        }
    }
}
