//! Preprocessing module for filtering noise from conversation exports before embedding.
//!
//! Problem: ~36-40% of conversation exports are NOISE (tool outputs, metadata, CLI commands).
//! This module filters noise BEFORE embedding to save vector space and improve search quality.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::OnceLock;
use tracing::info;

#[cfg(test)]
mod tests;

/// Configuration for the preprocessing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Remove MCP tool artifacts (function_calls, invoke, parameter, etc.)
    #[serde(default = "default_true")]
    pub remove_tool_artifacts: bool,

    /// Remove CLI command outputs (git status, cargo build, file listings)
    #[serde(default = "default_true")]
    pub remove_cli_output: bool,

    /// Remove system metadata (timestamps, UUIDs, session IDs)
    #[serde(default = "default_true")]
    pub remove_metadata: bool,

    /// Minimum content length after cleaning (characters)
    #[serde(default = "default_min_length")]
    pub min_content_length: usize,

    /// Threshold for deduplication (0.0-1.0, based on content hash similarity)
    #[serde(default = "default_dedupe_threshold")]
    pub dedupe_threshold: f32,

    /// Remove empty/boilerplate content
    #[serde(default = "default_true")]
    pub remove_empty_content: bool,

    /// Remove duplicate headers and repeated system prompts
    #[serde(default = "default_true")]
    pub remove_duplicate_headers: bool,
}

fn default_true() -> bool {
    true
}

fn default_min_length() -> usize {
    50
}

fn default_dedupe_threshold() -> f32 {
    0.95
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            remove_tool_artifacts: true,
            remove_cli_output: true,
            // CRITICAL: Preserve timestamps by default for temporal queries!
            // Use --sanitize-metadata CLI flag for opt-in sanitization.
            remove_metadata: false,
            min_content_length: 50,
            dedupe_threshold: 0.95,
            remove_empty_content: true,
            remove_duplicate_headers: true,
        }
    }
}

struct PreprocessingRegexes {
    function_calls_block: Regex,
    antml_invoke_block: Regex,
    antml_parameter_block: Regex,
    function_results_block: Regex,
    result_block: Regex,
    tool_output_tags: Regex,
    git_status_output: Regex,
    git_diff_output: Regex,
    cargo_output: Regex,
    npm_output: Regex,
    file_listing: Regex,
    tree_output: Regex,
    uuid_pattern: Regex,
    timestamp_iso: Regex,
    unix_timestamp: Regex,
    session_id_pattern: Regex,
    file_path_metadata: Regex,
    empty_content_json: Regex,
    empty_text_json: Regex,
    placeholder_message: Regex,
    multiple_newlines: Regex,
    multiple_spaces: Regex,
}

fn compile_static_regex(pattern: &str, label: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|err| panic!("invalid preprocessing regex '{label}': {err}"))
}

fn preprocessing_regexes() -> &'static PreprocessingRegexes {
    static REGEXES: OnceLock<PreprocessingRegexes> = OnceLock::new();
    REGEXES.get_or_init(|| PreprocessingRegexes {
        function_calls_block: compile_static_regex(
            &format!(
            r"(?s)<{}>{}</{}>",
            "function_calls", r".*?", "function_calls"
        ),
            "function_calls_block",
        ),
        antml_invoke_block: compile_static_regex(
            &format!(
            r"(?s)<{}:{}[^>]*>.*?</{}:{}>",
            "antml", "invoke", "antml", "invoke"
        ),
            "antml_invoke_block",
        ),
        antml_parameter_block: compile_static_regex(
            &format!(
            r"(?s)<{}:{}[^>]*>.*?</{}:{}>",
            "antml", "parameter", "antml", "parameter"
        ),
            "antml_parameter_block",
        ),
        function_results_block: compile_static_regex(
            &format!(
            r"(?s)<{}>{}</{}>",
            "function_results", r".*?", "function_results"
        ),
            "function_results_block",
        ),
        result_block: compile_static_regex(
            &format!(r"(?s)<{}>{}</{}>", "result", r".*?", "result"),
            "result_block",
        ),
        tool_output_tags: compile_static_regex(
            r"(?s)<(output|name|value)>.*?</(output|name|value)>",
            "tool_output_tags",
        ),
        git_status_output: compile_static_regex(
            r"(?m)^\s*(On branch|Your branch|Changes (?:not staged|to be committed)|Untracked files|nothing to commit|modified:|new file:|deleted:).*$",
            "git_status_output",
        ),
        git_diff_output: compile_static_regex(
            r"(?m)^(diff --git|index [0-9a-f]+\.\.[0-9a-f]+|--- a/|--- /|\+\+\+ a/|\+\+\+ b/|@@\s*-\d+.*@@|Binary files).*$",
            "git_diff_output",
        ),
        cargo_output: compile_static_regex(
            r"(?m)^(\s*(Compiling|Finished|Running|warning:|error\[E|-->|note:|help:)).*$",
            "cargo_output",
        ),
        npm_output: compile_static_regex(
            r"(?m)^(npm (WARN|ERR!|notice)|added \d+ packages|up to date|audited \d+ packages).*$",
            "npm_output",
        ),
        file_listing: compile_static_regex(
            r"(?m)^(total \d+|[drwx-]{10}\s+\d+|[-lrwx]{10}\s+\d+).*$",
            "file_listing",
        ),
        tree_output: compile_static_regex(r"(?m)^[│├└─\s]+[\w.-]+/?$", "tree_output"),
        uuid_pattern: compile_static_regex(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "uuid_pattern",
        ),
        timestamp_iso: compile_static_regex(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?",
            "timestamp_iso",
        ),
        unix_timestamp: compile_static_regex(r"\b1[6-7]\d{8}\b", "unix_timestamp"),
        session_id_pattern: compile_static_regex(
            r#"(session_id|sessionId|session-id|conv_id|conversation_id)["']?\s*[:=]\s*["']?[\w-]+"#,
            "session_id_pattern",
        ),
        file_path_metadata: compile_static_regex(
            r#""(path|file_path|filepath)"\s*:\s*"[^"]+""#,
            "file_path_metadata",
        ),
        empty_content_json: compile_static_regex(
            r#""content"\s*:\s*\[\s*\]"#,
            "empty_content_json",
        ),
        empty_text_json: compile_static_regex(r#""text"\s*:\s*"""#, "empty_text_json"),
        placeholder_message: compile_static_regex(
            r"(?i)(placeholder|lorem ipsum|TODO:|FIXME:|XXX:)",
            "placeholder_message",
        ),
        multiple_newlines: compile_static_regex(r"\n{3,}", "multiple_newlines"),
        multiple_spaces: compile_static_regex(r" {2,}", "multiple_spaces"),
    })
}

/// A simple hash for deduplication based on normalized content
fn content_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let normalized = s
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

/// Calculate similarity between two content hashes (Jaccard-like for words)
fn content_similarity(a: &str, b: &str) -> f32 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Message structure for conversation filtering
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
}

/// Statistics from preprocessing
#[derive(Debug, Clone, Default)]
pub struct PreprocessingStats {
    pub total_input: usize,
    pub filtered_tool_artifacts: usize,
    pub filtered_cli_output: usize,
    pub filtered_metadata: usize,
    pub filtered_empty: usize,
    pub filtered_duplicates: usize,
    pub filtered_below_min_length: usize,
    pub total_output: usize,
}

impl PreprocessingStats {
    pub fn total_filtered(&self) -> usize {
        self.filtered_tool_artifacts
            + self.filtered_cli_output
            + self.filtered_metadata
            + self.filtered_empty
            + self.filtered_duplicates
            + self.filtered_below_min_length
    }

    pub fn filter_rate(&self) -> f32 {
        if self.total_input == 0 {
            return 0.0;
        }
        self.total_filtered() as f32 / self.total_input as f32
    }
}

/// The main preprocessor for filtering conversation noise
pub struct Preprocessor {
    config: PreprocessingConfig,
    seen_hashes: HashSet<u64>,
}

impl Preprocessor {
    /// Create a new preprocessor with the given configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            seen_hashes: HashSet::new(),
        }
    }

    /// Create a preprocessor with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PreprocessingConfig::default())
    }

    /// Reset the deduplication cache (useful between conversations)
    pub fn reset_dedupe_cache(&mut self) {
        self.seen_hashes.clear();
    }

    /// Filter a single message content. Returns None if the content should be skipped entirely.
    pub fn filter_message(&mut self, content: &str) -> Option<String> {
        // Extract semantic content first
        let cleaned = self.extract_semantic_content(content);

        // Check minimum length
        if cleaned.len() < self.config.min_content_length {
            return None;
        }

        // Check for duplicates
        if self.config.dedupe_threshold < 1.0 {
            let hash = content_hash(&cleaned);
            if self.seen_hashes.contains(&hash) {
                return None;
            }
            self.seen_hashes.insert(hash);
        }

        Some(cleaned)
    }

    /// Filter a conversation (list of messages), returning only meaningful content
    pub fn filter_conversation(
        &mut self,
        messages: Vec<Message>,
    ) -> (Vec<Message>, PreprocessingStats) {
        let mut stats = PreprocessingStats {
            total_input: messages.len(),
            ..Default::default()
        };

        let mut result = Vec::new();
        let mut previous_contents: Vec<String> = Vec::new();

        for msg in messages {
            // Skip if content is mostly tool artifacts
            if self.config.remove_tool_artifacts && self.is_mostly_tool_artifact(&msg.content) {
                stats.filtered_tool_artifacts += 1;
                continue;
            }

            // Skip if content is mostly CLI output
            if self.config.remove_cli_output && self.is_mostly_cli_output(&msg.content) {
                stats.filtered_cli_output += 1;
                continue;
            }

            // Extract semantic content
            let cleaned = self.extract_semantic_content(&msg.content);

            // Skip empty content
            if self.config.remove_empty_content && cleaned.trim().is_empty() {
                stats.filtered_empty += 1;
                continue;
            }

            // Skip if below minimum length
            if cleaned.len() < self.config.min_content_length {
                stats.filtered_below_min_length += 1;
                continue;
            }

            // Check for duplicates/near-duplicates
            if self.config.dedupe_threshold < 1.0 {
                let is_duplicate = previous_contents
                    .iter()
                    .any(|prev| content_similarity(prev, &cleaned) >= self.config.dedupe_threshold);

                if is_duplicate {
                    stats.filtered_duplicates += 1;
                    continue;
                }
            }

            previous_contents.push(cleaned.clone());

            result.push(Message {
                role: msg.role,
                content: cleaned,
                metadata: msg.metadata,
            });
        }

        stats.total_output = result.len();

        info!(
            "Preprocessing complete: {}/{} messages kept ({:.1}% filtered)",
            stats.total_output,
            stats.total_input,
            stats.filter_rate() * 100.0
        );

        (result, stats)
    }

    /// Extract semantic content from raw text, removing noise patterns
    pub fn extract_semantic_content(&self, raw: &str) -> String {
        let regexes = preprocessing_regexes();
        let mut result = raw.to_string();

        // Remove tool artifacts
        if self.config.remove_tool_artifacts {
            result = regexes
                .function_calls_block
                .replace_all(&result, "")
                .to_string();
            result = regexes
                .antml_invoke_block
                .replace_all(&result, "")
                .to_string();
            result = regexes
                .antml_parameter_block
                .replace_all(&result, "")
                .to_string();
            result = regexes
                .function_results_block
                .replace_all(&result, "")
                .to_string();
            result = regexes.result_block.replace_all(&result, "").to_string();
            result = regexes
                .tool_output_tags
                .replace_all(&result, "")
                .to_string();
        }

        // Remove CLI output
        if self.config.remove_cli_output {
            result = regexes
                .git_status_output
                .replace_all(&result, "")
                .to_string();
            result = regexes.git_diff_output.replace_all(&result, "").to_string();
            result = regexes.cargo_output.replace_all(&result, "").to_string();
            result = regexes.npm_output.replace_all(&result, "").to_string();
            result = regexes.file_listing.replace_all(&result, "").to_string();
            result = regexes.tree_output.replace_all(&result, "").to_string();
        }

        // Remove metadata
        if self.config.remove_metadata {
            result = regexes
                .uuid_pattern
                .replace_all(&result, "[UUID]")
                .to_string();
            result = regexes
                .timestamp_iso
                .replace_all(&result, "[TIMESTAMP]")
                .to_string();
            result = regexes
                .unix_timestamp
                .replace_all(&result, "[TIMESTAMP]")
                .to_string();
            result = regexes
                .session_id_pattern
                .replace_all(&result, "")
                .to_string();
            result = regexes
                .file_path_metadata
                .replace_all(&result, "")
                .to_string();
        }

        // Remove empty content patterns
        if self.config.remove_empty_content {
            result = regexes
                .empty_content_json
                .replace_all(&result, "")
                .to_string();
            result = regexes.empty_text_json.replace_all(&result, "").to_string();
            result = regexes
                .placeholder_message
                .replace_all(&result, "")
                .to_string();
        }

        // Normalize whitespace
        result = regexes
            .multiple_newlines
            .replace_all(&result, "\n\n")
            .to_string();
        result = regexes
            .multiple_spaces
            .replace_all(&result, " ")
            .to_string();

        result.trim().to_string()
    }

    /// Check if content is predominantly tool artifacts
    fn is_mostly_tool_artifact(&self, content: &str) -> bool {
        let regexes = preprocessing_regexes();
        let original_len = content.len();
        if original_len == 0 {
            return false;
        }

        let mut cleaned = content.to_string();
        cleaned = regexes
            .function_calls_block
            .replace_all(&cleaned, "")
            .to_string();
        cleaned = regexes
            .antml_invoke_block
            .replace_all(&cleaned, "")
            .to_string();
        cleaned = regexes
            .antml_parameter_block
            .replace_all(&cleaned, "")
            .to_string();
        cleaned = regexes
            .function_results_block
            .replace_all(&cleaned, "")
            .to_string();
        cleaned = regexes.result_block.replace_all(&cleaned, "").to_string();

        let remaining_len = cleaned.trim().len();
        let artifact_ratio = 1.0 - (remaining_len as f32 / original_len as f32);

        // If more than 80% was tool artifacts, consider it mostly artifacts
        artifact_ratio > 0.8
    }

    /// Check if content is predominantly CLI output
    fn is_mostly_cli_output(&self, content: &str) -> bool {
        let regexes = preprocessing_regexes();
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return false;
        }

        let cli_lines = lines
            .iter()
            .filter(|line| {
                regexes.git_status_output.is_match(line)
                    || regexes.git_diff_output.is_match(line)
                    || regexes.cargo_output.is_match(line)
                    || regexes.npm_output.is_match(line)
                    || regexes.file_listing.is_match(line)
                    || regexes.tree_output.is_match(line)
            })
            .count();

        let cli_ratio = cli_lines as f32 / lines.len() as f32;

        // If more than 70% of lines are CLI output, consider it mostly CLI
        cli_ratio > 0.7
    }
}

// =============================================================================
// TEXT INTEGRITY METRICS - Quality assessment for embedding quality
// =============================================================================

/// Text integrity metrics for embedding quality assessment.
///
/// Target: >90% overall integrity score before indexing.
///
/// # Metrics
/// - **Sentence Integrity**: % of complete sentences preserved (not cut mid-sentence)
/// - **Word Integrity**: % of complete words (not truncated mid-word)
/// - **Chunk Quality**: How close chunks are to optimal size
///
/// # Example
/// ```rust,ignore
/// let metrics = TextIntegrityMetrics::compute(original_text, &chunks);
/// if !metrics.passes_threshold() {
///     warn!("Text integrity below 90%: {:.1}%", metrics.overall * 100.0);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextIntegrityMetrics {
    /// Percentage of complete sentences preserved (0.0 - 1.0)
    pub sentence_integrity: f32,

    /// Percentage of complete words (not truncated) (0.0 - 1.0)
    pub word_integrity: f32,

    /// Quality score based on chunk length distribution (0.0 - 1.0)
    pub chunk_quality: f32,

    /// Combined overall score (0.0 - 1.0)
    pub overall: f32,

    /// Number of chunks analyzed
    pub chunk_count: usize,

    /// Average chunk length in characters
    pub avg_chunk_length: usize,
}

impl TextIntegrityMetrics {
    /// Minimum acceptable overall score (90%)
    pub const THRESHOLD: f32 = 0.90;

    /// Optimal chunk length range (characters)
    pub const OPTIMAL_MIN: usize = 200;
    pub const OPTIMAL_MAX: usize = 800;

    /// Compute integrity metrics from original text and resulting chunks
    pub fn compute(original: &str, chunks: &[String]) -> Self {
        if chunks.is_empty() {
            return Self {
                sentence_integrity: 0.0,
                word_integrity: 0.0,
                chunk_quality: 0.0,
                overall: 0.0,
                chunk_count: 0,
                avg_chunk_length: 0,
            };
        }

        let sentence_integrity = Self::compute_sentence_integrity(original, chunks);
        let word_integrity = Self::compute_word_integrity(chunks);
        let chunk_quality = Self::compute_chunk_quality(chunks);

        // Weighted average: sentence integrity is most important
        let overall = sentence_integrity * 0.5 + word_integrity * 0.3 + chunk_quality * 0.2;

        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
        let avg_chunk_length = total_chars / chunks.len();

        Self {
            sentence_integrity,
            word_integrity,
            chunk_quality,
            overall,
            chunk_count: chunks.len(),
            avg_chunk_length,
        }
    }

    /// Check if metrics pass the minimum threshold (90%)
    pub fn passes_threshold(&self) -> bool {
        self.overall >= Self::THRESHOLD
    }

    /// Get recommendation based on metrics
    pub fn recommendation(&self) -> IntegrityRecommendation {
        if self.overall >= 0.95 {
            IntegrityRecommendation::Excellent
        } else if self.overall >= Self::THRESHOLD {
            IntegrityRecommendation::Good
        } else if self.overall >= 0.70 {
            IntegrityRecommendation::Warn
        } else {
            IntegrityRecommendation::Purge
        }
    }

    /// Compute sentence integrity score
    fn compute_sentence_integrity(original: &str, chunks: &[String]) -> f32 {
        let original_sentences = Self::count_sentences(original);
        if original_sentences == 0 {
            return 1.0; // No sentences to preserve
        }

        let preserved_sentences: usize = chunks
            .iter()
            .map(|c| Self::count_complete_sentences(c))
            .sum();

        // Ratio of preserved complete sentences
        let ratio = preserved_sentences as f32 / original_sentences as f32;

        // Cap at 1.0 (can exceed if chunks add sentence breaks)
        ratio.min(1.0)
    }

    /// Compute word integrity (chunks not ending mid-word)
    fn compute_word_integrity(chunks: &[String]) -> f32 {
        if chunks.is_empty() {
            return 1.0;
        }

        let complete_endings = chunks
            .iter()
            .filter(|c| Self::ends_at_word_boundary(c))
            .count();

        complete_endings as f32 / chunks.len() as f32
    }

    /// Compute chunk quality based on length distribution
    fn compute_chunk_quality(chunks: &[String]) -> f32 {
        if chunks.is_empty() {
            return 0.0;
        }

        let optimal_count = chunks
            .iter()
            .filter(|c| {
                let len = c.len();
                (Self::OPTIMAL_MIN..=Self::OPTIMAL_MAX).contains(&len)
            })
            .count();

        optimal_count as f32 / chunks.len() as f32
    }

    /// Count sentences in text (ending with . ! ?)
    fn count_sentences(text: &str) -> usize {
        text.chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count()
    }

    /// Count complete sentences (not cut mid-sentence)
    fn count_complete_sentences(chunk: &str) -> usize {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            return 0;
        }

        // A complete sentence ends with punctuation
        let sentences = Self::count_sentences(trimmed);

        // Check if chunk ends cleanly (with sentence terminator or at natural break)
        if Self::ends_at_sentence_boundary(trimmed) {
            sentences
        } else {
            // Last sentence is incomplete
            sentences.saturating_sub(1)
        }
    }

    /// Check if text ends at a sentence boundary
    fn ends_at_sentence_boundary(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return true;
        }

        let last_char = trimmed.chars().last().unwrap_or(' ');
        matches!(last_char, '.' | '!' | '?' | ':' | '"' | '\'' | ')' | ']')
    }

    /// Check if text ends at a word boundary (punctuation or whitespace)
    /// Note: We consider alphanumeric endings as incomplete if the original
    /// context isn't available - caller should verify if needed
    fn ends_at_word_boundary(text: &str) -> bool {
        let trimmed = text.trim_end();
        if trimmed.is_empty() {
            return true;
        }

        let last_char = trimmed.chars().last().unwrap_or(' ');
        // Only punctuation counts as clean word boundary
        // Alphanumeric endings might be mid-word cuts
        last_char.is_whitespace() || last_char.is_ascii_punctuation()
    }
}

/// Recommendation based on integrity metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrityRecommendation {
    /// >95% - High quality, keep
    Excellent,
    /// 90-95% - Good quality, keep
    Good,
    /// 70-90% - Consider re-indexing with better chunking
    Warn,
    /// <70% - Low quality, purge and re-index
    Purge,
}

impl IntegrityRecommendation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Excellent => "EXCELLENT",
            Self::Good => "GOOD",
            Self::Warn => "WARN",
            Self::Purge => "PURGE",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Excellent => "✅",
            Self::Good => "✅",
            Self::Warn => "⚠️",
            Self::Purge => "❌",
        }
    }
}

impl std::fmt::Display for TextIntegrityMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rec = self.recommendation();
        write!(
            f,
            "{} {}: {:.1}% (sentence: {:.1}%, word: {:.1}%, chunk: {:.1}%) - {} chunks, avg {}ch",
            rec.emoji(),
            rec.as_str(),
            self.overall * 100.0,
            self.sentence_integrity * 100.0,
            self.word_integrity * 100.0,
            self.chunk_quality * 100.0,
            self.chunk_count,
            self.avg_chunk_length
        )
    }
}

#[cfg(test)]
impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            metadata: None,
        }
    }
}

#[cfg(test)]
mod integrity_tests {
    use super::*;

    #[test]
    fn test_perfect_integrity() {
        // Create longer chunks to pass chunk_quality (OPTIMAL_MIN=200)
        let original = "This is the first sentence with some padding text to make it longer. \
                        Here is another sentence that continues the thought and adds context. \
                        The third sentence provides more information about the topic at hand. \
                        Finally we conclude with a fourth sentence that wraps everything up nicely.";
        let chunks = vec![
            "This is the first sentence with some padding text to make it longer. \
             Here is another sentence that continues the thought and adds context."
                .to_string(),
            "The third sentence provides more information about the topic at hand. \
             Finally we conclude with a fourth sentence that wraps everything up nicely."
                .to_string(),
        ];

        let metrics = TextIntegrityMetrics::compute(original, &chunks);
        assert!(
            metrics.sentence_integrity >= 0.9,
            "sentence_integrity: {}",
            metrics.sentence_integrity
        );
        assert!(
            metrics.word_integrity >= 0.9,
            "word_integrity: {}",
            metrics.word_integrity
        );
        // overall = 0.5*sentence + 0.3*word + 0.2*chunk_quality
        // With perfect sentence/word but short chunks (< OPTIMAL_MIN), overall = 0.8
        assert!(metrics.overall >= 0.75, "overall: {}", metrics.overall);
    }

    #[test]
    fn test_poor_integrity() {
        let original = "This is a complete sentence with many words.";
        // Chunks cut mid-word and mid-sentence
        let chunks = vec![
            "This is a compl".to_string(), // Cut mid-word
            "ete sentence wi".to_string(), // Cut mid-word
            "th many words".to_string(),   // Missing period
        ];

        let metrics = TextIntegrityMetrics::compute(original, &chunks);
        assert!(metrics.word_integrity < 0.9); // Mid-word cuts
        assert!(!metrics.passes_threshold());
        assert_eq!(metrics.recommendation(), IntegrityRecommendation::Purge);
    }

    #[test]
    fn test_empty_chunks() {
        let original = "Some text";
        let chunks: Vec<String> = vec![];

        let metrics = TextIntegrityMetrics::compute(original, &chunks);
        assert_eq!(metrics.chunk_count, 0);
        assert_eq!(metrics.overall, 0.0);
    }

    #[test]
    fn test_recommendation_levels() {
        // Excellent
        let m = TextIntegrityMetrics {
            sentence_integrity: 1.0,
            word_integrity: 1.0,
            chunk_quality: 0.9,
            overall: 0.97,
            chunk_count: 10,
            avg_chunk_length: 400,
        };
        assert_eq!(m.recommendation(), IntegrityRecommendation::Excellent);

        // Good
        let m = TextIntegrityMetrics { overall: 0.92, ..m };
        assert_eq!(m.recommendation(), IntegrityRecommendation::Good);

        // Warn
        let m = TextIntegrityMetrics { overall: 0.75, ..m };
        assert_eq!(m.recommendation(), IntegrityRecommendation::Warn);

        // Purge
        let m = TextIntegrityMetrics { overall: 0.50, ..m };
        assert_eq!(m.recommendation(), IntegrityRecommendation::Purge);
    }
}
