//! Query intent detection and routing for intelligent search.
//!
//! This module implements query analysis to determine the best search strategy
//! based on the user's intent. Different query types benefit from different
//! approaches:
//!
//! - **Temporal** queries ("when did X happen") need date filtering and timestamp priority
//! - **Structural** queries ("what imports X") should delegate to loctree
//! - **Semantic** queries ("similar to X") work best with pure vector search
//! - **Exact** queries (quoted strings) need BM25 keyword matching
//! - **Hybrid** queries (default) combine vector + BM25 for best results
//!
//! # Example
//!
//! ```rust
//! use rmcp_memex::query::{QueryIntent, detect_intent, QueryRouter, RoutingDecision};
//!
//! let intent = detect_intent("when did we buy host-a");
//! assert!(matches!(intent, QueryIntent::Temporal));
//!
//! let router = QueryRouter::new();
//! let decision = router.route("what imports this module");
//! // decision.delegate_to_loctree would be true
//! ```

use std::collections::HashSet;
use std::sync::OnceLock;

/// Represents the detected intent of a user query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// Temporal queries asking about when something happened.
    /// Keywords: when, date, time, before, after, since, until, ago
    /// Best approach: Add date filtering, prioritize timestamps
    Temporal,

    /// Structural queries about code/file relationships.
    /// Keywords: import, depend, call, reference, use, module
    /// Best approach: Delegate to loctree or suggest loctree command
    Structural,

    /// Semantic queries looking for conceptually similar content.
    /// Keywords: similar, like, related, about, explain, understand
    /// Best approach: Pure vector search for semantic similarity
    Semantic,

    /// Exact match queries with quoted strings or specific keywords.
    /// Pattern: Contains quoted strings like "exact phrase"
    /// Best approach: BM25 keyword matching
    Exact,

    /// Hybrid queries that don't fit other categories (default).
    /// Best approach: Fusion of vector + BM25 search
    Hybrid,
}

impl std::fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryIntent::Temporal => write!(f, "temporal"),
            QueryIntent::Structural => write!(f, "structural"),
            QueryIntent::Semantic => write!(f, "semantic"),
            QueryIntent::Exact => write!(f, "exact"),
            QueryIntent::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Configuration for temporal query handling.
#[derive(Debug, Clone)]
pub struct TemporalHints {
    /// Detected date references in the query
    pub date_references: Vec<String>,
    /// Whether to prioritize recent results
    pub prefer_recent: bool,
    /// Suggested date range filter (start, end) as ISO strings
    pub suggested_range: Option<(String, String)>,
}

/// Suggestions for loctree delegation.
#[derive(Debug, Clone)]
pub struct LoctreeSuggestion {
    /// The loctree command to run
    pub command: String,
    /// Human-readable explanation
    pub explanation: String,
}

/// The routing decision made by QueryRouter.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Primary detected intent
    pub intent: QueryIntent,
    /// Secondary intents (queries can have multiple aspects)
    pub secondary_intents: Vec<QueryIntent>,
    /// Confidence score 0.0-1.0
    pub confidence: f32,
    /// Should this query be delegated to loctree?
    pub delegate_to_loctree: bool,
    /// Loctree command suggestion if applicable
    pub loctree_suggestion: Option<LoctreeSuggestion>,
    /// Temporal hints if temporal intent detected
    pub temporal_hints: Option<TemporalHints>,
    /// Recommended search mode
    pub recommended_mode: RecommendedSearchMode,
}

/// Recommended search configuration based on intent.
#[derive(Debug, Clone)]
pub struct RecommendedSearchMode {
    /// Primary search mode
    pub mode: SearchModeRecommendation,
    /// BM25 weight (0.0-1.0) for hybrid search
    pub bm25_weight: f32,
    /// Vector weight (0.0-1.0) for hybrid search
    pub vector_weight: f32,
    /// Whether to boost exact matches
    pub boost_exact: bool,
}

/// Search mode recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchModeRecommendation {
    /// Pure vector search
    Vector,
    /// Pure BM25 keyword search
    Bm25,
    /// Hybrid fusion of both
    Hybrid,
}

// Keyword sets for intent detection
fn temporal_keywords() -> &'static HashSet<&'static str> {
    static TEMPORAL_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    TEMPORAL_KEYWORDS.get_or_init(|| {
        [
            "when",
            "date",
            "time",
            "before",
            "after",
            "since",
            "until",
            "ago",
            "yesterday",
            "today",
            "tomorrow",
            "last",
            "next",
            "month",
            "week",
            "year",
            "kiedy",
            "data",
            "czas",
            "przed",
            "po",
            "wczoraj",
            "dzisiaj",
        ]
        .into_iter()
        .collect()
    })
}

fn structural_keywords() -> &'static HashSet<&'static str> {
    static STRUCTURAL_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STRUCTURAL_KEYWORDS.get_or_init(|| {
        [
            "import",
            "imports",
            "depend",
            "depends",
            "dependency",
            "dependencies",
            "call",
            "calls",
            "reference",
            "references",
            "use",
            "uses",
            "module",
            "file",
            "function",
            "class",
            "struct",
            "interface",
            "where",
            "defined",
            "definition",
            "who",
        ]
        .into_iter()
        .collect()
    })
}

fn semantic_keywords() -> &'static HashSet<&'static str> {
    static SEMANTIC_KEYWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SEMANTIC_KEYWORDS.get_or_init(|| {
        [
            "similar",
            "like",
            "related",
            "about",
            "explain",
            "understand",
            "meaning",
            "concept",
            "idea",
            "topic",
            "theme",
            "podobny",
            "zwiazany",
        ]
        .into_iter()
        .collect()
    })
}

/// Detect the primary intent of a query.
///
/// This is a fast heuristic-based detection. For more nuanced analysis,
/// use `QueryRouter::route()` which provides confidence scores and
/// secondary intents.
///
/// # Arguments
///
/// * `query` - The search query string
///
/// # Returns
///
/// The detected `QueryIntent`
///
/// # Example
///
/// ```rust
/// use rmcp_memex::query::detect_intent;
///
/// let intent = detect_intent("when did we buy host-a");
/// // Returns QueryIntent::Temporal
/// ```
pub fn detect_intent(query: &str) -> QueryIntent {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Check for quoted strings first (exact match intent)
    if query.contains('"') || query.contains('\'') {
        return QueryIntent::Exact;
    }

    // Count keyword matches for each category
    let temporal_count = words
        .iter()
        .filter(|w| temporal_keywords().contains(*w))
        .count();
    let structural_count = words
        .iter()
        .filter(|w| structural_keywords().contains(*w))
        .count();
    let semantic_count = words
        .iter()
        .filter(|w| semantic_keywords().contains(*w))
        .count();

    // Check for date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
    let has_date_pattern = lower.contains("2024")
        || lower.contains("2025")
        || lower.contains("2023")
        || lower.chars().filter(|c| *c == '-').count() >= 2
            && lower.chars().filter(|c| c.is_ascii_digit()).count() >= 4;

    // Determine winner
    if temporal_count > 0 || has_date_pattern {
        return QueryIntent::Temporal;
    }
    if structural_count >= 2
        || (structural_count > 0 && (lower.contains("who ") || lower.contains("what ")))
    {
        return QueryIntent::Structural;
    }
    if semantic_count > 0 {
        return QueryIntent::Semantic;
    }

    // Default to hybrid for complex queries
    QueryIntent::Hybrid
}

/// Query router that provides detailed routing decisions.
#[derive(Debug, Clone)]
pub struct QueryRouter {
    /// Weight threshold for temporal detection
    temporal_threshold: f32,
    /// Weight threshold for structural detection
    structural_threshold: f32,
}

impl Default for QueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryRouter {
    /// Create a new QueryRouter with default settings.
    pub fn new() -> Self {
        Self {
            temporal_threshold: 0.3,
            structural_threshold: 0.4,
        }
    }

    /// Route a query and return a detailed routing decision.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    ///
    /// # Returns
    ///
    /// A `RoutingDecision` containing intent, confidence, and recommendations
    pub fn route(&self, query: &str) -> RoutingDecision {
        let primary_intent = detect_intent(query);
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let word_count = words.len().max(1) as f32;

        // Calculate scores for each intent
        let temporal_score = words
            .iter()
            .filter(|w| temporal_keywords().contains(*w))
            .count() as f32
            / word_count;
        let structural_score = words
            .iter()
            .filter(|w| structural_keywords().contains(*w))
            .count() as f32
            / word_count;
        let semantic_score = words
            .iter()
            .filter(|w| semantic_keywords().contains(*w))
            .count() as f32
            / word_count;

        // Calculate confidence based on keyword density
        let confidence = match primary_intent {
            QueryIntent::Temporal => (temporal_score * 2.0 + 0.3).min(1.0),
            QueryIntent::Structural => (structural_score * 2.0 + 0.3).min(1.0),
            QueryIntent::Semantic => (semantic_score * 2.0 + 0.3).min(1.0),
            QueryIntent::Exact => 0.9, // High confidence for quoted strings
            QueryIntent::Hybrid => 0.5, // Default confidence
        };

        // Detect secondary intents
        let mut secondary_intents = Vec::new();
        if primary_intent != QueryIntent::Temporal && temporal_score >= self.temporal_threshold {
            secondary_intents.push(QueryIntent::Temporal);
        }
        if primary_intent != QueryIntent::Structural
            && structural_score >= self.structural_threshold
        {
            secondary_intents.push(QueryIntent::Structural);
        }

        // Check if we should delegate to loctree
        let delegate_to_loctree = primary_intent == QueryIntent::Structural;
        let loctree_suggestion = if delegate_to_loctree {
            Some(self.suggest_loctree_command(query))
        } else {
            None
        };

        // Extract temporal hints if applicable
        let temporal_hints = if primary_intent == QueryIntent::Temporal
            || secondary_intents.contains(&QueryIntent::Temporal)
        {
            Some(self.extract_temporal_hints(query))
        } else {
            None
        };

        // Determine recommended search mode
        let recommended_mode = self.recommend_search_mode(primary_intent);

        RoutingDecision {
            intent: primary_intent,
            secondary_intents,
            confidence,
            delegate_to_loctree,
            loctree_suggestion,
            temporal_hints,
            recommended_mode,
        }
    }

    /// Suggest a loctree command for structural queries.
    fn suggest_loctree_command(&self, query: &str) -> LoctreeSuggestion {
        let lower = query.to_lowercase();

        if lower.contains("who imports") || lower.contains("what imports") {
            // Extract the target (rough heuristic)
            let target = query
                .split_whitespace()
                .skip_while(|w| !w.eq_ignore_ascii_case("imports"))
                .nth(1)
                .unwrap_or("TARGET");

            LoctreeSuggestion {
                command: format!("loctree query --kind who-imports --target {}", target),
                explanation: format!(
                    "Find all files that import '{}' using loctree's dependency graph",
                    target
                ),
            }
        } else if lower.contains("where") && lower.contains("defined") {
            let target = query
                .split_whitespace()
                .find(|w| {
                    !["where", "is", "the", "defined", "definition", "of"]
                        .contains(&w.to_lowercase().as_str())
                })
                .unwrap_or("SYMBOL");

            LoctreeSuggestion {
                command: format!("loctree find --name {}", target),
                explanation: format!(
                    "Find where '{}' is defined using loctree's symbol index",
                    target
                ),
            }
        } else if lower.contains("depend") {
            LoctreeSuggestion {
                command: "loctree impact --file PATH".to_string(),
                explanation: "Use loctree impact analysis to see what depends on a file"
                    .to_string(),
            }
        } else {
            LoctreeSuggestion {
                command: "loctree for_ai".to_string(),
                explanation: "Get an AI-optimized overview of the project structure".to_string(),
            }
        }
    }

    /// Extract temporal hints from a query.
    fn extract_temporal_hints(&self, query: &str) -> TemporalHints {
        let lower = query.to_lowercase();
        let mut date_references = Vec::new();

        // Look for year references
        for year in &["2023", "2024", "2025"] {
            if lower.contains(year) {
                date_references.push(year.to_string());
            }
        }

        // Look for relative time references
        let relative_terms = ["yesterday", "today", "last week", "last month", "ago"];
        for term in relative_terms {
            if lower.contains(term) {
                date_references.push(term.to_string());
            }
        }

        let prefer_recent = lower.contains("recent")
            || lower.contains("latest")
            || lower.contains("newest")
            || lower.contains("last");

        TemporalHints {
            date_references,
            prefer_recent,
            suggested_range: None, // Could be enhanced to parse actual date ranges
        }
    }

    /// Recommend search mode based on intent.
    fn recommend_search_mode(&self, intent: QueryIntent) -> RecommendedSearchMode {
        match intent {
            QueryIntent::Temporal => RecommendedSearchMode {
                mode: SearchModeRecommendation::Hybrid,
                bm25_weight: 0.5,
                vector_weight: 0.5,
                boost_exact: true, // Boost exact date matches
            },
            QueryIntent::Structural => RecommendedSearchMode {
                mode: SearchModeRecommendation::Bm25,
                bm25_weight: 0.8,
                vector_weight: 0.2,
                boost_exact: true,
            },
            QueryIntent::Semantic => RecommendedSearchMode {
                mode: SearchModeRecommendation::Vector,
                bm25_weight: 0.1,
                vector_weight: 0.9,
                boost_exact: false,
            },
            QueryIntent::Exact => RecommendedSearchMode {
                mode: SearchModeRecommendation::Bm25,
                bm25_weight: 1.0,
                vector_weight: 0.0,
                boost_exact: true,
            },
            QueryIntent::Hybrid => RecommendedSearchMode {
                mode: SearchModeRecommendation::Hybrid,
                bm25_weight: 0.4,
                vector_weight: 0.6,
                boost_exact: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_intent_detection() {
        assert_eq!(
            detect_intent("when did we buy host-a"),
            QueryIntent::Temporal
        );
        assert_eq!(
            detect_intent("what happened last week"),
            QueryIntent::Temporal
        );
        assert_eq!(
            detect_intent("meetings from 2024-11-15"),
            QueryIntent::Temporal
        );
        assert_eq!(detect_intent("before the deadline"), QueryIntent::Temporal);
    }

    #[test]
    fn test_structural_intent_detection() {
        assert_eq!(
            detect_intent("what imports this module"),
            QueryIntent::Structural
        );
        assert_eq!(
            detect_intent("who uses the function parse_json"),
            QueryIntent::Structural
        );
        assert_eq!(
            detect_intent("file dependencies for main.rs"),
            QueryIntent::Structural
        );
    }

    #[test]
    fn test_semantic_intent_detection() {
        assert_eq!(
            detect_intent("find similar discussions"),
            QueryIntent::Semantic
        );
        assert_eq!(
            detect_intent("topics related to rust"),
            QueryIntent::Semantic
        );
        assert_eq!(
            detect_intent("explain the concept of ownership"),
            QueryIntent::Semantic
        );
    }

    #[test]
    fn test_exact_intent_detection() {
        assert_eq!(detect_intent(r#"find "exact phrase""#), QueryIntent::Exact);
        assert_eq!(detect_intent("search 'host-a'"), QueryIntent::Exact);
    }

    #[test]
    fn test_hybrid_default() {
        assert_eq!(detect_intent("host-a mac studio"), QueryIntent::Hybrid);
        assert_eq!(detect_intent("how to configure memex"), QueryIntent::Hybrid);
    }

    #[test]
    fn test_router_confidence() {
        let router = QueryRouter::new();

        let decision = router.route("when exactly did we buy host-a in 2024");
        assert_eq!(decision.intent, QueryIntent::Temporal);
        assert!(decision.confidence > 0.5);

        let decision = router.route("random query here");
        assert_eq!(decision.intent, QueryIntent::Hybrid);
        assert!(decision.confidence <= 0.6);
    }

    #[test]
    fn test_loctree_suggestion() {
        let router = QueryRouter::new();

        let decision = router.route("who imports main.rs");
        assert!(decision.delegate_to_loctree);
        assert!(decision.loctree_suggestion.is_some());
        let suggestion = decision.loctree_suggestion.unwrap();
        assert!(suggestion.command.contains("who-imports"));
    }

    #[test]
    fn test_temporal_hints() {
        let router = QueryRouter::new();

        let decision = router.route("what happened in 2024");
        assert!(decision.temporal_hints.is_some());
        let hints = decision.temporal_hints.unwrap();
        assert!(hints.date_references.contains(&"2024".to_string()));
    }

    #[test]
    fn test_search_mode_recommendations() {
        let router = QueryRouter::new();

        let decision = router.route("find similar code");
        assert_eq!(
            decision.recommended_mode.mode,
            SearchModeRecommendation::Vector
        );
        assert!(decision.recommended_mode.vector_weight > 0.8);

        let decision = router.route(r#"search "exact match""#);
        assert_eq!(
            decision.recommended_mode.mode,
            SearchModeRecommendation::Bm25
        );
        assert!(decision.recommended_mode.bm25_weight > 0.9);
    }
}
