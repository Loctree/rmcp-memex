//! Query routing and intent detection for intelligent search.
//!
//! This module provides query analysis capabilities to determine the best
//! search strategy based on user intent. It helps route queries to the
//! appropriate search backend or external tools like loctree.
//!
//! # Architecture
//!
//! ```text
//! Query -> [Intent Detection] -> QueryIntent
//!       -> [Query Router] -> RoutingDecision
//!                            ├── Temporal → Date filtering + timestamp priority
//!                            ├── Structural → Delegate to loctree
//!                            ├── Semantic → Pure vector search
//!                            ├── Exact → BM25 keyword match
//!                            └── Hybrid → Vector + BM25 fusion
//! ```
//!
//! # Usage
//!
//! ```rust
//! use rmcp_memex::query::{detect_intent, QueryIntent, QueryRouter};
//!
//! // Quick intent detection
//! let intent = detect_intent("when did we buy dragon");
//! assert!(matches!(intent, QueryIntent::Temporal));
//!
//! // Full routing with recommendations
//! let router = QueryRouter::new();
//! let decision = router.route("what imports main.rs");
//! if let Some(suggestion) = decision.loctree_suggestion {
//!     println!("Suggested: {}", suggestion.command);
//! }
//! ```

pub mod router;

pub use router::{
    LoctreeSuggestion, QueryIntent, QueryRouter, RecommendedSearchMode, RoutingDecision,
    SearchModeRecommendation, TemporalHints, detect_intent,
};
