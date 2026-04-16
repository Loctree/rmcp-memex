use anyhow::{Result, anyhow};
use pdf_extract;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

use crate::{
    embeddings::MLXBridge,
    preprocessing::{PreprocessingConfig, Preprocessor},
    search::BM25Index,
    storage::{ChromaDocument, CrossStoreRecoveryBatch, CrossStoreRecoveryStatus, StorageManager},
};

// Async pipeline module for concurrent indexing
pub mod pipeline;
pub mod structured;
pub use pipeline::{
    Chunk, EmbeddedChunk, FileContent, PipelineConfig, PipelineEvent, PipelineGovernorConfig,
    PipelineResult, PipelineSnapshot, PipelineStats, run_pipeline,
};

const DEFAULT_NAMESPACE: &str = "rag";

/// Storage batch size - write to LanceDB every N documents to avoid RAM explosion
/// and enable crash recovery for large file indexing.
const STORAGE_BATCH_SIZE: usize = 100;

// =============================================================================
// ONION SLICE ARCHITECTURE
// =============================================================================
//
// The onion-like slice architecture creates hierarchical embeddings:
//   OUTER  (~100 chars) - Keywords, topic, participants
//   MIDDLE (~300 chars) - Key points, decisions, summary
//   INNER  (~600 chars) - Detailed context, quotes, reasoning
//   CORE   (full text)  - Complete original content
//
// Philosophy: "Minimum info -> Maximum navigation paths"
// Search returns OUTER slices by default; user drills down as needed.
// =============================================================================

/// Layer in the onion-like slice hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SliceLayer {
    /// ~100 chars - Keywords, topic, "What is this about?"
    Outer = 1,
    /// ~300 chars - Key points, summary, "What happened?"
    Middle = 2,
    /// ~600 chars - Detailed context, "How did it happen?"
    Inner = 3,
    /// Full content - Complete original text
    Core = 4,
}

impl SliceLayer {
    /// Target character count for this layer
    pub fn target_chars(&self) -> usize {
        match self {
            SliceLayer::Outer => 100,
            SliceLayer::Middle => 300,
            SliceLayer::Inner => 600,
            SliceLayer::Core => usize::MAX,
        }
    }

    /// Convert to u8 for storage
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Convert from u8
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(SliceLayer::Outer),
            2 => Some(SliceLayer::Middle),
            3 => Some(SliceLayer::Inner),
            4 => Some(SliceLayer::Core),
            _ => None,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SliceLayer::Outer => "outer",
            SliceLayer::Middle => "middle",
            SliceLayer::Inner => "inner",
            SliceLayer::Core => "core",
        }
    }
}

impl std::fmt::Display for SliceLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A slice in the onion hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionSlice {
    /// Unique ID for this slice (hash-based)
    pub id: String,
    /// Layer in the hierarchy
    pub layer: SliceLayer,
    /// The slice content
    pub content: String,
    /// Parent slice ID (None for Core)
    pub parent_id: Option<String>,
    /// Children slice IDs (empty for Outer)
    pub children_ids: Vec<String>,
    /// Extracted keywords for this slice
    pub keywords: Vec<String>,
}

impl OnionSlice {
    /// Generate a deterministic ID from content hash
    pub fn generate_id(content: &str, layer: SliceLayer) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        layer.as_u8().hash(&mut hasher);
        format!("slice_{:016x}_{}", hasher.finish(), layer.name())
    }
}

/// Slicing mode for document indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceMode {
    /// Hierarchical onion slicing with all 4 layers (default)
    #[default]
    Onion,
    /// Fast onion: only outer + core layers (2x faster, good for large datasets)
    OnionFast,
    /// Traditional flat chunking (backward compatible)
    Flat,
}

impl std::str::FromStr for SliceMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "onion" => Ok(SliceMode::Onion),
            "onion-fast" | "fast" => Ok(SliceMode::OnionFast),
            "flat" => Ok(SliceMode::Flat),
            other => Err(format!(
                "Invalid slice mode: '{}'. Use 'onion', 'onion-fast', or 'flat'",
                other
            )),
        }
    }
}

/// Result of indexing operation with deduplication
#[derive(Debug, Clone)]
pub enum IndexResult {
    /// Content was indexed successfully
    Indexed {
        /// Number of chunks/slices indexed
        chunks_indexed: usize,
        /// Content hash for the indexed content
        content_hash: String,
        /// Time spent in embedding calls (ms)
        embedder_ms: Option<u64>,
        /// Estimated total tokens for this content
        tokens_estimated: Option<usize>,
    },
    /// Content was skipped because it already exists (exact-match duplicate)
    Skipped {
        /// Reason for skipping
        reason: String,
        /// Content hash that was found as duplicate
        content_hash: String,
    },
}

impl IndexResult {
    /// Check if content was indexed
    pub fn was_indexed(&self) -> bool {
        matches!(self, IndexResult::Indexed { .. })
    }

    #[deprecated(note = "use was_indexed")]
    pub fn is_indexed(&self) -> bool {
        self.was_indexed()
    }

    /// Check if content was skipped
    pub fn is_skipped(&self) -> bool {
        matches!(self, IndexResult::Skipped { .. })
    }

    /// Get the content hash
    pub fn content_hash(&self) -> &str {
        match self {
            IndexResult::Indexed { content_hash, .. } => content_hash,
            IndexResult::Skipped { content_hash, .. } => content_hash,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CrossStoreRecoveryState {
    Clean,
    Divergent,
    RolledBack,
    Stale,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossStoreRecoveryBatchReport {
    pub batch_id: String,
    pub namespace: String,
    pub created_at: String,
    pub state: CrossStoreRecoveryState,
    pub status: CrossStoreRecoveryStatus,
    pub document_count: usize,
    pub lance_documents: usize,
    pub bm25_documents: usize,
    pub missing_bm25_ids: Vec<String>,
    pub missing_lance_ids: Vec<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct CrossStoreRecoveryReport {
    pub recovery_dir: String,
    pub pending_batches: usize,
    pub clean_batches: usize,
    pub divergent_batches: usize,
    pub rolled_back_batches: usize,
    pub stale_batches: usize,
    pub documents_examined: usize,
    pub documents_missing_bm25: usize,
    pub documents_missing_lance: usize,
    pub repaired_documents: usize,
    pub skipped_documents: usize,
    pub cleared_batches: usize,
    pub batches_repaired: usize,
    pub batches: Vec<CrossStoreRecoveryBatchReport>,
}

impl CrossStoreRecoveryReport {
    pub fn is_clean(&self) -> bool {
        self.pending_batches == 0
            || (self.divergent_batches == 0
                && self.rolled_back_batches == 0
                && self.stale_batches == 0)
    }
}

async fn run_cross_store_recovery(
    storage: &StorageManager,
    bm25: &BM25Index,
    namespace: Option<&str>,
    execute: bool,
) -> Result<CrossStoreRecoveryReport> {
    let recovery_dir = storage.cross_store_recovery_dir();
    let mut report = CrossStoreRecoveryReport {
        recovery_dir: recovery_dir.display().to_string(),
        ..Default::default()
    };

    let batches = storage
        .list_cross_store_recovery_batches()?
        .into_iter()
        .filter(|batch| {
            namespace.is_none_or(|expected| {
                batch
                    .documents
                    .iter()
                    .any(|document| document.namespace == expected)
            })
        })
        .collect::<Vec<_>>();
    report.pending_batches = batches.len();

    if batches.is_empty() {
        return Ok(report);
    }

    let mut lance_cache: HashMap<String, HashMap<String, ChromaDocument>> = HashMap::new();
    let mut bm25_cache: HashMap<String, HashSet<String>> = HashMap::new();

    for batch in batches {
        let namespace_name = batch
            .documents
            .first()
            .map(|document| document.namespace.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let lance_documents = if let Some(documents) = lance_cache.get(&namespace_name) {
            documents
        } else {
            let documents = storage
                .get_all_in_namespace(&namespace_name)
                .await?
                .into_iter()
                .map(|document| (document.id.clone(), document))
                .collect::<HashMap<_, _>>();
            lance_cache.insert(namespace_name.clone(), documents);
            lance_cache
                .get(&namespace_name)
                .expect("just inserted lance cache")
        };

        let bm25_documents = if let Some(ids) = bm25_cache.get(&namespace_name) {
            ids
        } else {
            let ids = bm25
                .document_keys(Some(&namespace_name))?
                .into_iter()
                .filter_map(|(doc_namespace, id)| (doc_namespace == namespace_name).then_some(id))
                .collect::<HashSet<_>>();
            bm25_cache.insert(namespace_name.clone(), ids);
            bm25_cache
                .get(&namespace_name)
                .expect("just inserted bm25 cache")
        };

        let mut missing_bm25_ids = Vec::new();
        let mut missing_lance_ids = Vec::new();
        let mut repair_documents = Vec::new();
        let mut lance_present = 0usize;
        let mut bm25_present = 0usize;

        for document_ref in &batch.documents {
            report.documents_examined += 1;

            if let Some(document) = lance_documents.get(&document_ref.id) {
                lance_present += 1;
                if bm25_documents.contains(&document_ref.id) {
                    bm25_present += 1;
                } else {
                    missing_bm25_ids.push(document_ref.id.clone());
                    repair_documents.push((
                        document.id.clone(),
                        document.namespace.clone(),
                        document.document.clone(),
                    ));
                }
            } else {
                missing_lance_ids.push(document_ref.id.clone());
            }
        }

        report.documents_missing_bm25 += missing_bm25_ids.len();
        report.documents_missing_lance += missing_lance_ids.len();

        let state = if !missing_bm25_ids.is_empty() {
            report.divergent_batches += 1;
            CrossStoreRecoveryState::Divergent
        } else if batch.status == CrossStoreRecoveryStatus::RolledBack {
            report.rolled_back_batches += 1;
            CrossStoreRecoveryState::RolledBack
        } else if !missing_lance_ids.is_empty() || lance_present == 0 {
            report.stale_batches += 1;
            CrossStoreRecoveryState::Stale
        } else {
            report.clean_batches += 1;
            CrossStoreRecoveryState::Clean
        };

        if execute {
            match state {
                CrossStoreRecoveryState::Divergent => {
                    bm25.add_documents(&repair_documents).await?;
                    if let Some(ids) = bm25_cache.get_mut(&namespace_name) {
                        for (id, _, _) in &repair_documents {
                            ids.insert(id.clone());
                        }
                    }
                    report.repaired_documents += repair_documents.len();
                    report.skipped_documents += missing_lance_ids.len();
                    report.batches_repaired += 1;
                    storage.clear_cross_store_recovery_batch(&batch.batch_id)?;
                    report.cleared_batches += 1;
                }
                CrossStoreRecoveryState::RolledBack
                | CrossStoreRecoveryState::Stale
                | CrossStoreRecoveryState::Clean => {
                    report.skipped_documents += missing_lance_ids.len();
                    storage.clear_cross_store_recovery_batch(&batch.batch_id)?;
                    report.cleared_batches += 1;
                }
            }
        }

        report.batches.push(CrossStoreRecoveryBatchReport {
            batch_id: batch.batch_id,
            namespace: namespace_name,
            created_at: batch.created_at,
            state,
            status: batch.status,
            document_count: batch.documents.len(),
            lance_documents: lance_present,
            bm25_documents: bm25_present,
            missing_bm25_ids,
            missing_lance_ids,
            last_error: batch.last_error,
        });
    }

    Ok(report)
}

pub async fn inspect_cross_store_recovery(
    storage: &StorageManager,
    bm25: &BM25Index,
    namespace: Option<&str>,
) -> Result<CrossStoreRecoveryReport> {
    run_cross_store_recovery(storage, bm25, namespace, false).await
}

pub async fn repair_cross_store_recovery(
    storage: &StorageManager,
    bm25: &BM25Index,
    namespace: Option<&str>,
) -> Result<CrossStoreRecoveryReport> {
    run_cross_store_recovery(storage, bm25, namespace, true).await
}

/// Compute SHA256 hash of content and return as hex string
pub fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    // Convert to hex string (64 chars for SHA256)
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Configuration for onion slicing
#[derive(Debug, Clone)]
pub struct OnionSliceConfig {
    /// Target size for outer layer (~100 chars)
    pub outer_target: usize,
    /// Target size for middle layer (~300 chars)
    pub middle_target: usize,
    /// Target size for inner layer (~600 chars)
    pub inner_target: usize,
    /// Minimum content length to apply onion slicing (below this, use single Core slice)
    pub min_content_for_slicing: usize,
}

impl Default for OnionSliceConfig {
    fn default() -> Self {
        Self {
            outer_target: 100,
            middle_target: 300,
            inner_target: 600,
            min_content_for_slicing: 200,
        }
    }
}

fn create_core_only_slice(content: &str) -> Vec<OnionSlice> {
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let keywords = extract_keywords(content, 5);
    vec![OnionSlice {
        id: core_id,
        layer: SliceLayer::Core,
        content: content.to_string(),
        parent_id: None,
        children_ids: vec![],
        keywords,
    }]
}

/// Create onion slices from content
///
/// Algorithm:
/// 1. Full content -> CORE slice
/// 2. Extract key sentences -> INNER slice (~600 chars)
/// 3. Summarize to key points -> MIDDLE slice (~300 chars)
/// 4. Extract keywords/topic -> OUTER slice (~100 chars)
pub fn create_onion_slices(
    content: &str,
    metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    if structured::is_structured_conversation(metadata) {
        return structured::create_structured_onion_slices(content, metadata, config);
    }

    let content = content.trim();

    // For very short content, keep at least Outer+Core for structured dialog turns
    // so outer-only search can still see short but meaningful conversational units.
    if content.len() < config.min_content_for_slicing {
        return create_core_only_slice(content);
    }

    let mut slices = Vec::with_capacity(4);

    // 1. CORE slice - full content
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    // 2. INNER slice - extract key sentences (~600 chars)
    let inner_content = extract_key_content(content, config.inner_target);
    let inner_id = OnionSlice::generate_id(&inner_content, SliceLayer::Inner);
    let inner_keywords = extract_keywords(&inner_content, 7);

    // 3. MIDDLE slice - summarize to key points (~300 chars)
    let middle_content = extract_key_content(&inner_content, config.middle_target);
    let middle_id = OnionSlice::generate_id(&middle_content, SliceLayer::Middle);
    let middle_keywords = extract_keywords(&middle_content, 5);

    // 4. OUTER slice - keywords and topic (~100 chars)
    let outer_content = create_outer_summary(&middle_content, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    // Build hierarchy with parent/children links
    slices.push(OnionSlice {
        id: outer_id.clone(),
        layer: SliceLayer::Outer,
        content: outer_content,
        parent_id: Some(middle_id.clone()),
        children_ids: vec![],
        keywords: outer_keywords,
    });

    slices.push(OnionSlice {
        id: middle_id.clone(),
        layer: SliceLayer::Middle,
        content: middle_content,
        parent_id: Some(inner_id.clone()),
        children_ids: vec![outer_id],
        keywords: middle_keywords,
    });

    slices.push(OnionSlice {
        id: inner_id.clone(),
        layer: SliceLayer::Inner,
        content: inner_content,
        parent_id: Some(core_id.clone()),
        children_ids: vec![middle_id],
        keywords: inner_keywords,
    });

    slices.push(OnionSlice {
        id: core_id.clone(),
        layer: SliceLayer::Core,
        content: content.to_string(),
        parent_id: None,
        children_ids: vec![inner_id],
        keywords: core_keywords,
    });

    slices
}

/// Create fast onion slices (outer + core only) - 2x faster than full onion
///
/// For bulk indexing where search quality can be slightly reduced.
/// Outer layer enables fast keyword-style search, Core provides full content.
pub fn create_onion_slices_fast(
    content: &str,
    metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    if structured::is_structured_conversation(metadata) {
        return structured::create_structured_onion_slices_fast(content, metadata, config);
    }

    let content = content.trim();

    // Fast mode keeps the same structured short-content behavior as full onion mode.
    if content.len() < config.min_content_for_slicing {
        return create_core_only_slice(content);
    }

    let mut slices = Vec::with_capacity(2);

    // CORE slice - full content
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    // OUTER slice - keywords and topic (~100 chars)
    // Derive from core directly (skip middle/inner)
    let outer_content = create_outer_summary(content, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    // Build minimal hierarchy
    slices.push(OnionSlice {
        id: outer_id.clone(),
        layer: SliceLayer::Outer,
        content: outer_content,
        parent_id: Some(core_id.clone()),
        children_ids: vec![],
        keywords: outer_keywords,
    });

    slices.push(OnionSlice {
        id: core_id,
        layer: SliceLayer::Core,
        content: content.to_string(),
        parent_id: None,
        children_ids: vec![outer_id],
        keywords: core_keywords,
    });

    slices
}

/// Extract keywords from text using simple TF-based extraction
fn extract_keywords(text: &str, max_keywords: usize) -> Vec<String> {
    use std::collections::HashMap;

    // Common stop words to filter out
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
        "who", "whom", "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "also", "now", "here", "there", "then", "once", "if", "into",
        "through", "during", "before", "after", "above", "below", "between", "under", "again",
        "further", "about", "out", "over", "up", "down", "off", "any", "because", "until", "while",
    ];

    let stop_set: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();

    // Tokenize and count word frequencies
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for raw in text.split_whitespace() {
        for token in tokenize_keyword_candidates(raw) {
            if token.len() >= 3
                && token.len() <= 30
                && !stop_set.contains(token.as_str())
                && !looks_like_session_token(&token)
            {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }
    }

    // Sort by frequency and take top N
    let mut words: Vec<_> = word_counts.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1));

    words
        .into_iter()
        .take(max_keywords)
        .map(|(word, _)| word)
        .collect()
}

fn tokenize_keyword_candidates(raw: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for segment in raw
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|segment| !segment.is_empty())
    {
        let compact: String = segment.chars().flat_map(|ch| ch.to_lowercase()).collect();
        let mut normalized = String::with_capacity(segment.len() * 2);
        let mut previous_is_lowercase = false;

        for ch in segment.chars() {
            if ch.is_ascii_uppercase() && previous_is_lowercase {
                normalized.push(' ');
            }

            normalized.push(ch.to_ascii_lowercase());
            previous_is_lowercase = ch.is_ascii_lowercase();
        }

        let segment_tokens = normalized
            .split_whitespace()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        tokens.extend(segment_tokens.iter().cloned());

        if segment_tokens.len() > 1
            && compact.len() >= 3
            && compact.len() <= 30
            && !tokens.iter().any(|token| token == &compact)
        {
            tokens.push(compact);
        }
    }

    tokens
}

fn looks_like_session_token(token: &str) -> bool {
    let hex_chars = token.chars().filter(|ch| ch.is_ascii_hexdigit()).count();
    let digit_chars = token.chars().filter(|ch| ch.is_ascii_digit()).count();
    let alpha_chars = token.chars().filter(|ch| ch.is_ascii_alphabetic()).count();

    token.len() > 12 && hex_chars == token.len()
        || digit_chars >= 6
        || (token.len() > 20 && alpha_chars < token.len() / 3)
}

/// Create short hash for document deduplication
fn hash_content(text: &str) -> String {
    let mut hash = compute_content_hash(text);
    hash.truncate(16);
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptRole {
    User,
    Assistant,
    Reasoning,
}

#[derive(Debug, Clone)]
struct StructuredDialogEntry {
    time: Option<String>,
    role: TranscriptRole,
    text: String,
}

#[derive(Debug, Default, Clone)]
struct MarkdownTranscriptTurn {
    start_time: Option<String>,
    end_time: Option<String>,
    user_segments: Vec<String>,
    assistant_segments: Vec<String>,
    reasoning_segments: Vec<String>,
}

impl MarkdownTranscriptTurn {
    fn is_empty(&self) -> bool {
        self.user_segments.is_empty()
            && self.assistant_segments.is_empty()
            && self.reasoning_segments.is_empty()
    }

    fn push(&mut self, role: TranscriptRole, time: &str, text: String) {
        if !time.is_empty() {
            if self.start_time.is_none() {
                self.start_time = Some(time.to_string());
            }
            self.end_time = Some(time.to_string());
        }

        let target = match role {
            TranscriptRole::User => &mut self.user_segments,
            TranscriptRole::Assistant => &mut self.assistant_segments,
            TranscriptRole::Reasoning => &mut self.reasoning_segments,
        };

        if target.last().is_some_and(|existing| existing == &text) {
            return;
        }
        target.push(text);
    }
}

fn parse_transcript_header(line: &str) -> Option<HashMap<String, String>> {
    let inner = line.trim().strip_prefix('[')?.strip_suffix(']')?.trim();

    if !inner.starts_with("project:") {
        return None;
    }

    let mut fields = HashMap::new();
    for segment in inner.split(" | ") {
        let (key, value) = segment.split_once(':')?;
        fields.insert(key.trim().to_string(), value.trim().to_string());
    }

    Some(fields)
}

fn parse_transcript_role(role: &str) -> Option<TranscriptRole> {
    match role.trim().to_ascii_lowercase().as_str() {
        "user" | "human" => Some(TranscriptRole::User),
        "assistant" | "bot" | "model" => Some(TranscriptRole::Assistant),
        "reasoning" | "analysis" | "thinking" | "tool" => Some(TranscriptRole::Reasoning),
        _ => None,
    }
}

fn parse_transcript_entry_line(line: &str) -> Option<(String, TranscriptRole, String)> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('[')?;
    let (time, rest) = rest.split_once(']')?;
    let is_timestamp = time.len() == 8
        && time.chars().enumerate().all(|(idx, ch)| match idx {
            2 | 5 => ch == ':',
            _ => ch.is_ascii_digit(),
        });
    if !is_timestamp {
        return None;
    }

    let (role, body) = rest.trim_start().split_once(':')?;
    let role = parse_transcript_role(role)?;

    Some((time.to_string(), role, body.trim_start().to_string()))
}

fn normalize_role_aware_turn(turn: &MarkdownTranscriptTurn) -> Option<String> {
    let mut sections = Vec::new();

    if !turn.user_segments.is_empty() {
        sections.push(format!("User request:\n{}", turn.user_segments.join("\n")));
    }
    if !turn.assistant_segments.is_empty() {
        sections.push(format!(
            "Assistant response:\n{}",
            turn.assistant_segments.join("\n")
        ));
    }
    if !turn.reasoning_segments.is_empty() {
        sections.push(format!(
            "Reasoning focus:\n{}",
            turn.reasoning_segments.join("\n")
        ));
    }

    if sections.is_empty() {
        return None;
    }

    Some(sections.join("\n\n"))
}

fn build_role_aware_turn_documents(
    entries: Vec<StructuredDialogEntry>,
    doc_prefix: &str,
    base_metadata: serde_json::Map<String, serde_json::Value>,
) -> Vec<(String, String, serde_json::Value)> {
    let mut turns = Vec::new();
    let mut current_turn = MarkdownTranscriptTurn::default();

    for entry in entries {
        let text = entry.text.trim();
        if text.is_empty() {
            continue;
        }

        if matches!(entry.role, TranscriptRole::User) && !current_turn.is_empty() {
            turns.push(current_turn);
            current_turn = MarkdownTranscriptTurn::default();
        }

        current_turn.push(
            entry.role,
            entry.time.as_deref().unwrap_or_default(),
            text.to_string(),
        );
    }

    if !current_turn.is_empty() {
        turns.push(current_turn);
    }

    let mut docs = Vec::new();
    for (idx, turn) in turns.into_iter().enumerate() {
        let Some(content) = normalize_role_aware_turn(&turn) else {
            continue;
        };

        if content.len() < 20 {
            continue;
        }

        let doc_id = format!("{doc_prefix}-{idx:04}-{}", hash_content(&content));
        let mut metadata = serde_json::Value::Object(base_metadata.clone());

        if let serde_json::Value::Object(ref mut map) = metadata {
            map.insert("turn_index".to_string(), json!(idx));
            map.insert("start_time".to_string(), json!(turn.start_time));
            map.insert("end_time".to_string(), json!(turn.end_time));
        }

        docs.push((doc_id, content, metadata));
    }

    docs
}

fn extract_markdown_transcript_documents(
    raw: &str,
    source_path: &std::path::Path,
) -> Option<Vec<(String, String, serde_json::Value)>> {
    let mut header = HashMap::new();
    let mut current_entry: Option<(String, TranscriptRole, String)> = None;
    let mut entries = Vec::new();
    let mut in_signals = false;
    let mut signal_lines = Vec::new();

    for line in raw.lines() {
        if header.is_empty()
            && let Some(parsed) = parse_transcript_header(line)
        {
            header = parsed;
            continue;
        }

        let trimmed = line.trim();
        if trimmed == "[signals]" {
            in_signals = true;
            continue;
        }
        if trimmed == "[/signals]" {
            in_signals = false;
            continue;
        }

        if let Some((time, role, body)) = parse_transcript_entry_line(line) {
            if let Some(entry) = current_entry.take() {
                entries.push(entry);
            }
            current_entry = Some((time, role, body));
            continue;
        }

        if in_signals {
            if !trimmed.is_empty() {
                signal_lines.push(trimmed.to_string());
            }
            continue;
        }

        if let Some((_, _, ref mut text)) = current_entry
            && !trimmed.is_empty()
        {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(trimmed);
        }
    }

    if let Some(entry) = current_entry.take() {
        entries.push(entry);
    }

    if header.is_empty() || entries.is_empty() {
        return None;
    }

    let project = header
        .get("project")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let agent = header
        .get("agent")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let date = header
        .get("date")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let source_name = source_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let transcript_id = hash_content(&source_path.to_string_lossy());
    let signals_summary = if signal_lines.is_empty() {
        None
    } else {
        Some(signal_lines.join("\n"))
    };

    let entries = entries
        .into_iter()
        .map(|(time, role, text)| StructuredDialogEntry {
            time: Some(time),
            role,
            text,
        })
        .collect();

    let mut metadata = serde_json::Map::new();
    metadata.insert("project".to_string(), json!(project));
    metadata.insert("agent".to_string(), json!(agent));
    metadata.insert("date".to_string(), json!(date));
    metadata.insert("source".to_string(), json!(source_name));
    metadata.insert("path".to_string(), json!(source_path.to_str()));
    metadata.insert("type".to_string(), json!("transcript_turn"));
    metadata.insert("format".to_string(), json!("markdown_transcript"));

    if let Some(summary) = signals_summary.as_ref() {
        metadata.insert("signals".to_string(), json!(summary));
    }

    let docs =
        build_role_aware_turn_documents(entries, &format!("mdturn-{transcript_id}"), metadata);

    if docs.is_empty() { None } else { Some(docs) }
}

/// Extract conversation documents from JSON with smart format detection.
/// Returns role-aware turn documents with proper metadata.
/// Handles: sessions format, ChatGPT export, generic messages array.
fn extract_conversation_documents(
    value: &serde_json::Value,
    source_path: &std::path::Path,
) -> Option<Vec<(String, String, serde_json::Value)>> {
    let obj = value.as_object()?;
    let source_name = source_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    fn extract_text_blocks(msg_obj: &serde_json::Map<String, serde_json::Value>) -> String {
        if let Some(text) = msg_obj.get("text").and_then(serde_json::Value::as_str) {
            return text.to_string();
        }
        if let Some(content) = msg_obj.get("content") {
            if let Some(text) = content.as_str() {
                return text.to_string();
            }
            if let Some(blocks) = content.as_array() {
                return blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(serde_json::Value::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
            }
        }
        String::new()
    }

    // Pattern 1: {sessions: [{info: {}, messages: [{role, text, timestamp}]}]}
    // From extract_session_essence.py output
    if let Some(serde_json::Value::Array(sessions)) = obj.get("sessions") {
        let project = obj
            .get("project")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let mut docs = Vec::new();
        for session in sessions {
            let session_obj = session.as_object()?;
            let session_id = session_obj
                .get("info")
                .and_then(|i| i.get("sessionId"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let session_short = &session_id[..session_id.len().min(8)];

            let mut entries = Vec::new();
            if let Some(serde_json::Value::Array(messages)) = session_obj.get("messages") {
                for msg in messages {
                    let msg_obj = match msg.as_object() {
                        Some(o) => o,
                        None => continue,
                    };
                    let Some(role) = msg_obj
                        .get("role")
                        .and_then(|v| v.as_str())
                        .and_then(parse_transcript_role)
                    else {
                        continue;
                    };

                    let text = msg_obj
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if text.len() < 20 {
                        continue;
                    }

                    let timestamp = msg_obj
                        .get("timestamp")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned);

                    entries.push(StructuredDialogEntry {
                        time: timestamp,
                        role,
                        text,
                    });
                }
            }

            let mut metadata = serde_json::Map::new();
            metadata.insert("session".to_string(), json!(session_short));
            metadata.insert("project".to_string(), json!(project));
            metadata.insert("source".to_string(), json!(source_name));
            metadata.insert("type".to_string(), json!("conversation"));
            metadata.insert("format".to_string(), json!("sessions"));

            let mut session_docs = build_role_aware_turn_documents(
                entries,
                &format!("sess-{session_short}"),
                metadata,
            );
            docs.append(&mut session_docs);
        }

        if !docs.is_empty() {
            tracing::info!(
                "Sessions format detected: {} -> {} turn documents",
                source_path.display(),
                docs.len()
            );
            return Some(docs);
        }
    }

    // Pattern 2: {uuid, name, messages: [{sender, text, created_at}]}
    // Claude.ai conversations export (conversations-merged.json)
    // This is handled at array level in extract_json_documents, but check for single conversation
    if let Some(serde_json::Value::Array(messages)) = obj.get("messages") {
        let conv_id = obj
            .get("uuid")
            .or_else(|| obj.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let conv_short = &conv_id[..conv_id.len().min(8)];
        let title = obj
            .get("name")
            .or_else(|| obj.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let looks_like_conversation = messages.iter().any(|m| {
            m.get("sender").is_some() || m.get("role").is_some() || m.get("author").is_some()
        });

        if looks_like_conversation {
            let mut entries = Vec::new();
            for msg in messages {
                let msg_obj = match msg.as_object() {
                    Some(o) => o,
                    None => continue,
                };

                let Some(role) = msg_obj
                    .get("sender")
                    .or_else(|| msg_obj.get("role"))
                    .or_else(|| msg_obj.get("author").and_then(|a| a.get("role")))
                    .and_then(|v| v.as_str())
                    .and_then(parse_transcript_role)
                else {
                    continue;
                };

                let text = extract_text_blocks(msg_obj).trim().to_string();
                if text.len() < 20 {
                    continue;
                }

                let timestamp = msg_obj
                    .get("created_at")
                    .or_else(|| msg_obj.get("timestamp"))
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned);

                entries.push(StructuredDialogEntry {
                    time: timestamp,
                    role,
                    text,
                });
            }

            let mut metadata = serde_json::Map::new();
            metadata.insert("conversation".to_string(), json!(conv_short));
            metadata.insert("title".to_string(), json!(title));
            metadata.insert("source".to_string(), json!(source_name));
            metadata.insert("type".to_string(), json!("conversation"));
            metadata.insert("format".to_string(), json!("claude_web"));

            let docs =
                build_role_aware_turn_documents(entries, &format!("conv-{conv_short}"), metadata);

            if !docs.is_empty() {
                tracing::info!(
                    "Conversation format detected: {} -> {} turn documents",
                    source_path.display(),
                    docs.len()
                );
                return Some(docs);
            }
        }
    }

    // Pattern 4: {uuid, chat_messages: [{sender, text, created_at}]}
    // Claude.ai conversations export format (conversations-merged.json)
    if let Some(serde_json::Value::Array(messages)) = obj.get("chat_messages") {
        let conv_id = obj
            .get("uuid")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let conv_short = &conv_id[..conv_id.len().min(8)];
        let title = obj.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let mut entries = Vec::new();
        for msg in messages {
            let msg_obj = match msg.as_object() {
                Some(o) => o,
                None => continue,
            };

            let Some(role) = msg_obj
                .get("sender")
                .and_then(|v| v.as_str())
                .and_then(parse_transcript_role)
            else {
                continue;
            };

            let text = msg_obj
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if text.len() < 20 {
                continue;
            }

            let timestamp = msg_obj
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned);

            entries.push(StructuredDialogEntry {
                time: timestamp,
                role,
                text,
            });
        }

        let mut metadata = serde_json::Map::new();
        metadata.insert("conversation".to_string(), json!(conv_short));
        metadata.insert("title".to_string(), json!(title));
        metadata.insert("source".to_string(), json!(source_name));
        metadata.insert("type".to_string(), json!("conversation"));
        metadata.insert("format".to_string(), json!("claude_web"));

        let docs =
            build_role_aware_turn_documents(entries, &format!("chat-{conv_short}"), metadata);

        if !docs.is_empty() {
            tracing::info!(
                "Claude.ai chat_messages format detected: {} -> {} turn documents",
                source_path.display(),
                docs.len()
            );
            return Some(docs);
        }
    }

    // Pattern 3: {mapping: {id: {message: {content: {parts: []}}}}}
    // ChatGPT export format
    if let Some(serde_json::Value::Object(mapping)) = obj.get("mapping") {
        let conv_id = obj
            .get("id")
            .or_else(|| obj.get("conversation_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let conv_short = &conv_id[..conv_id.len().min(8)];
        let title = obj.get("title").and_then(|v| v.as_str()).unwrap_or("");

        let mut entries: Vec<_> = mapping.iter().collect();
        entries.sort_by(|a, b| {
            let time_a =
                a.1.get("message")
                    .and_then(|m| m.get("create_time"))
                    .and_then(|t| t.as_f64())
                    .unwrap_or(0.0);
            let time_b =
                b.1.get("message")
                    .and_then(|m| m.get("create_time"))
                    .and_then(|t| t.as_f64())
                    .unwrap_or(0.0);
            time_a
                .partial_cmp(&time_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut dialog_entries = Vec::new();
        for (_node_id, node) in entries {
            let message = match node.get("message") {
                Some(m) => m,
                None => continue,
            };

            let Some(role) = message
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(|v| v.as_str())
                .and_then(parse_transcript_role)
            else {
                continue;
            };

            let text = message
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default()
                .trim()
                .to_string();
            if text.len() < 20 {
                continue;
            }

            let timestamp = message
                .get("create_time")
                .and_then(|t| t.as_f64())
                .and_then(|ts| chrono::DateTime::from_timestamp(ts as i64, 0))
                .map(|dt| dt.to_rfc3339());

            dialog_entries.push(StructuredDialogEntry {
                time: timestamp,
                role,
                text,
            });
        }

        let mut metadata = serde_json::Map::new();
        metadata.insert("conversation".to_string(), json!(conv_short));
        metadata.insert("title".to_string(), json!(title));
        metadata.insert("source".to_string(), json!(source_name));
        metadata.insert("type".to_string(), json!("conversation"));
        metadata.insert("format".to_string(), json!("chatgpt"));

        let docs =
            build_role_aware_turn_documents(dialog_entries, &format!("gpt-{conv_short}"), metadata);

        if !docs.is_empty() {
            tracing::info!(
                "ChatGPT format detected: {} -> {} turn documents",
                source_path.display(),
                docs.len()
            );
            return Some(docs);
        }
    }

    None
}

/// Extract meaningful text content from a JSON element (object or value).
/// Handles common patterns: messages, conversations, entities, generic objects.
fn extract_json_element_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            let mut parts = Vec::new();

            // Priority fields for conversation/chat data
            for key in [
                "content",
                "text",
                "message",
                "summary",
                "description",
                "body",
            ] {
                if let Some(serde_json::Value::String(s)) = map.get(key)
                    && !s.is_empty()
                {
                    parts.push(s.clone());
                }
            }

            // Handle role-based messages (user/assistant)
            if let Some(serde_json::Value::String(role)) = map.get("role")
                && let Some(content) = map.get("content")
            {
                match content {
                    serde_json::Value::String(s) => {
                        parts.push(format!("{}: {}", role, s));
                    }
                    serde_json::Value::Array(arr) => {
                        // Content blocks (Claude format)
                        for item in arr {
                            if let serde_json::Value::Object(block) = item
                                && let Some(serde_json::Value::String(t)) = block.get("text")
                            {
                                parts.push(format!("{}: {}", role, t));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Handle nested messages array
            if let Some(serde_json::Value::Array(messages)) = map.get("messages") {
                for msg in messages.iter().take(50) {
                    // Limit to avoid huge outputs
                    let msg_content = extract_json_element_content(msg);
                    if !msg_content.is_empty() && msg_content.len() > 10 {
                        parts.push(msg_content);
                    }
                }
            }

            // Handle chat_messages (ChatGPT format)
            if let Some(serde_json::Value::Array(messages)) = map.get("chat_messages") {
                for msg in messages.iter().take(50) {
                    let msg_content = extract_json_element_content(msg);
                    if !msg_content.is_empty() && msg_content.len() > 10 {
                        parts.push(msg_content);
                    }
                }
            }

            // Handle entity memories
            if let Some(serde_json::Value::String(name)) = map.get("name")
                && let Some(serde_json::Value::Array(obs)) = map.get("observations")
            {
                let observations: Vec<String> = obs
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .take(10)
                    .collect();
                if !observations.is_empty() {
                    parts.push(format!("{}: {}", name, observations.join("; ")));
                }
            }

            // Title/name for context
            for key in ["title", "name", "uuid", "id"] {
                if let Some(serde_json::Value::String(s)) = map.get(key) {
                    if !s.is_empty() && parts.iter().all(|p| !p.contains(s)) {
                        parts.insert(0, format!("[{}]", s));
                    }
                    break;
                }
            }

            if parts.is_empty() {
                // Fallback: serialize the whole object (limited)
                serde_json::to_string(value)
                    .unwrap_or_default()
                    .chars()
                    .take(5000)
                    .collect()
            } else {
                parts.join("\n")
            }
        }
        serde_json::Value::Array(arr) => {
            // For arrays, extract each element
            arr.iter()
                .take(20)
                .map(extract_json_element_content)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => value.to_string(),
    }
}

/// Detect the type of JSON element for metadata
fn detect_json_element_type(value: &serde_json::Value) -> &'static str {
    if let serde_json::Value::Object(map) = value {
        // Check for conversation patterns
        if map.contains_key("chat_messages") || map.contains_key("mapping") {
            return "conversation";
        }
        if map.contains_key("messages") && map.contains_key("sessions") {
            return "session";
        }
        if map.contains_key("role") && map.contains_key("content") {
            return "message";
        }
        if map.contains_key("observations") && map.contains_key("name") {
            return "entity";
        }
        if map.contains_key("messages") {
            return "thread";
        }
        "object"
    } else if value.is_array() {
        "array"
    } else if value.is_string() {
        "text"
    } else {
        "value"
    }
}

/// Extract key content from text, targeting a specific character count
/// Uses sentence-based extraction to maintain coherence
fn extract_key_content(text: &str, target_chars: usize) -> String {
    if text.len() <= target_chars {
        return text.to_string();
    }

    // Split into sentences (simple heuristic)
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        // Fallback: truncate with word boundary
        return truncate_at_word_boundary(text, target_chars);
    }

    // Score sentences by position and keyword density
    let keywords = extract_keywords(text, 10);
    let keyword_set: std::collections::HashSet<&str> =
        keywords.iter().map(|s| s.as_str()).collect();

    let mut scored_sentences: Vec<(usize, f32, &str)> = sentences
        .iter()
        .enumerate()
        .map(|(idx, sentence)| {
            let mut score = 0.0_f32;

            // Position score: first and last sentences are often important
            if idx == 0 {
                score += 2.0;
            } else if idx == sentences.len() - 1 {
                score += 1.5;
            }

            // Keyword density score
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let keyword_count = words
                .iter()
                .filter(|w| {
                    let cleaned: String = w
                        .chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                        .to_lowercase();
                    keyword_set.contains(cleaned.as_str())
                })
                .count();

            if !words.is_empty() {
                score += (keyword_count as f32 / words.len() as f32) * 3.0;
            }

            // Length penalty for very short sentences
            if sentence.len() < 20 {
                score -= 0.5;
            }

            (idx, score, *sentence)
        })
        .collect();

    // Sort by score (descending)
    scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Collect sentences until we hit target length, maintaining original order
    let mut selected_indices: Vec<usize> = Vec::new();
    let mut total_len = 0;

    for (idx, _, sentence) in &scored_sentences {
        let sentence_len = sentence.len() + 2; // +2 for ". "
        if total_len + sentence_len > target_chars && !selected_indices.is_empty() {
            break;
        }
        selected_indices.push(*idx);
        total_len += sentence_len;
    }

    // Sort by original position to maintain text flow
    selected_indices.sort();

    // Reconstruct text
    let result: Vec<&str> = selected_indices
        .iter()
        .filter_map(|&idx| sentences.get(idx).copied())
        .collect();

    if result.is_empty() {
        truncate_at_word_boundary(text, target_chars)
    } else {
        result.join(". ") + "."
    }
}

/// Create an outer summary from middle content and keywords
fn create_outer_summary(middle_content: &str, keywords: &[String], target_chars: usize) -> String {
    // Start with top keywords
    let keyword_prefix = if !keywords.is_empty() {
        format!(
            "[{}] ",
            keywords
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::new()
    };

    let remaining_chars = target_chars.saturating_sub(keyword_prefix.len());

    // Take first sentence or truncate
    let first_sentence = middle_content
        .split(['.', '!', '?'])
        .next()
        .unwrap_or(middle_content)
        .trim();

    let summary = if first_sentence.len() <= remaining_chars {
        first_sentence.to_string()
    } else {
        truncate_at_word_boundary(first_sentence, remaining_chars)
    };

    format!("{}{}", keyword_prefix, summary)
}

/// Truncate text at a word boundary (UTF-8 safe)
fn truncate_at_word_boundary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    // Get byte index of max_chars-th character (UTF-8 safe)
    let byte_idx = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());

    let truncated = &text[..byte_idx];

    // Find the last space before cutoff
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

pub struct RAGPipeline {
    mlx_bridge: Arc<Mutex<MLXBridge>>,
    storage: Arc<StorageManager>,
    bm25_writer: Option<Arc<BM25Index>>,
}

impl RAGPipeline {
    /// Create new RAGPipeline with MLXBridge (required, no fallback!)
    pub async fn new(
        mlx_bridge: Arc<Mutex<MLXBridge>>,
        storage: Arc<StorageManager>,
    ) -> Result<Self> {
        Self::new_with_bm25(mlx_bridge, storage, None).await
    }

    pub async fn new_with_bm25(
        mlx_bridge: Arc<Mutex<MLXBridge>>,
        storage: Arc<StorageManager>,
        bm25_writer: Option<Arc<BM25Index>>,
    ) -> Result<Self> {
        Ok(Self {
            mlx_bridge,
            storage,
            bm25_writer,
        })
    }

    pub fn storage_manager(&self) -> Arc<StorageManager> {
        self.storage.clone()
    }

    /// Refresh storage to see new data written by other processes
    pub async fn refresh(&self) -> Result<()> {
        self.storage.refresh().await
    }

    /// Best-effort cross-store write:
    /// 1. persist a recovery ledger for the batch,
    /// 2. write LanceDB,
    /// 3. sync BM25,
    /// 4. clear the ledger on success.
    ///
    /// Same-process BM25 failures still trigger Lance rollback, but this is not
    /// crash-safe atomicity. If the process dies between steps, operators must
    /// reconcile with the explicit `repair-writes` surface rather than assuming
    /// rollback already made both stores truthful.
    async fn persist_documents(&self, documents: Vec<ChromaDocument>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        let mut unique_documents = Vec::with_capacity(documents.len());
        let mut seen_ids: HashSet<(String, String)> = HashSet::new();
        let mut seen_hashes: HashSet<(String, String)> = HashSet::new();

        for mut document in documents {
            if let Value::Object(ref mut map) = document.metadata {
                map.entry("indexed_at".to_string())
                    .or_insert_with(|| json!(chrono::Utc::now().to_rfc3339()));
            }

            let id_key = (document.namespace.clone(), document.id.clone());
            if !seen_ids.insert(id_key) {
                continue;
            }

            if let Some(hash) = document.content_hash.as_ref() {
                let hash_key = (document.namespace.clone(), hash.clone());
                if !seen_hashes.insert(hash_key) {
                    continue;
                }
            }

            unique_documents.push(document);
        }

        let documents = self
            .filter_documents_against_store(unique_documents)
            .await?;
        if documents.is_empty() {
            return Ok(());
        }

        let bm25_documents: Vec<(String, String, String)> = documents
            .iter()
            .map(|doc| (doc.id.clone(), doc.namespace.clone(), doc.document.clone()))
            .collect();
        let inserted_ids: Vec<(String, String)> = documents
            .iter()
            .map(|doc| (doc.namespace.clone(), doc.id.clone()))
            .collect();
        let mut recovery_batch = self
            .bm25_writer
            .as_ref()
            .map(|_| CrossStoreRecoveryBatch::from_documents(&documents));
        let recovery_path = if let Some(batch) = recovery_batch.as_ref() {
            Some(self.storage.persist_cross_store_recovery_batch(batch)?)
        } else {
            None
        };

        if let Err(error) = self.storage.add_to_store(documents).await {
            if let (Some(batch), Some(path)) = (recovery_batch.as_mut(), recovery_path.as_ref()) {
                batch.last_error = Some(format!("Lance write failed: {error}"));
                let _ = self.storage.update_cross_store_recovery_batch(batch);
                return Err(anyhow!(
                    "Lance write failed while cross-store recovery ledger was active at {}: {}. \
                     This path is not crash-safe; run `rmcp-memex repair-writes --execute` to reconcile Lance/BM25 truth after investigating the primary failure.",
                    path.display(),
                    error
                ));
            }
            return Err(error);
        }

        if let Some(bm25_writer) = &self.bm25_writer
            && let Err(error) = bm25_writer.add_documents(&bm25_documents).await
        {
            let mut rollback_failures = 0usize;
            for (namespace, id) in &inserted_ids {
                if self.storage.delete_document(namespace, id).await.is_err() {
                    rollback_failures += 1;
                }
            }

            if let Some(batch) = recovery_batch.as_mut() {
                batch.status = if rollback_failures == 0 {
                    CrossStoreRecoveryStatus::RolledBack
                } else {
                    CrossStoreRecoveryStatus::Pending
                };
                batch.last_error = Some(format!(
                    "BM25 write failed after Lance persist: {error}; lance_rollback_failures={rollback_failures}"
                ));
                let _ = self.storage.update_cross_store_recovery_batch(batch);
            }

            return Err(anyhow!(
                "BM25 write failed after Lance persist: {}. Recovery ledger preserved at {}. \
                 Same-process rollback attempted for {} documents and {} rollback operations failed. \
                 This remains recoverable through `rmcp-memex repair-writes --execute`, but it is not the same as crash-safe cross-store atomicity.",
                error,
                recovery_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<not-recorded>".to_string()),
                inserted_ids.len(),
                rollback_failures
            ));
        }

        if let Some(batch) = recovery_batch {
            self.storage
                .clear_cross_store_recovery_batch(&batch.batch_id)?;
        }
        Ok(())
    }

    async fn filter_documents_against_store(
        &self,
        documents: Vec<ChromaDocument>,
    ) -> Result<Vec<ChromaDocument>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let mut hashes_by_namespace: HashMap<String, Vec<String>> = HashMap::new();
        for document in &documents {
            if let Some(hash) = document.content_hash.as_ref() {
                hashes_by_namespace
                    .entry(document.namespace.clone())
                    .or_default()
                    .push(hash.clone());
            }
        }

        let mut allowed_hashes: HashMap<String, HashSet<String>> = HashMap::new();
        for (namespace, hashes) in hashes_by_namespace {
            let hashes = self
                .storage
                .filter_existing_hashes(&namespace, &hashes)
                .await?;
            allowed_hashes.insert(
                namespace,
                hashes.into_iter().cloned().collect::<HashSet<_>>(),
            );
        }

        Ok(documents
            .into_iter()
            .filter(|document| match document.content_hash.as_ref() {
                None => true,
                Some(hash) => allowed_hashes
                    .get(&document.namespace)
                    .map(|hashes| hashes.contains(hash))
                    .unwrap_or(true),
            })
            .collect())
    }

    async fn clear_namespace_from_indices(&self, namespace: &str) -> Result<usize> {
        let deleted = self.storage.delete_namespace_documents(namespace).await?;

        if let Some(bm25_writer) = &self.bm25_writer {
            bm25_writer.delete_namespace_term(namespace).await?;
        }

        Ok(deleted)
    }

    async fn load_memory_family(&self, namespace: &str, id: &str) -> Result<Vec<ChromaDocument>> {
        let docs = self.storage.get_all_in_namespace(namespace).await?;
        Ok(docs
            .into_iter()
            .filter(|doc| {
                doc.id == id
                    || doc
                        .metadata
                        .get("original_id")
                        .and_then(|value| value.as_str())
                        .is_some_and(|original_id| original_id == id)
            })
            .collect())
    }

    async fn delete_memory_family(&self, namespace: &str, id: &str) -> Result<usize> {
        let family = self.load_memory_family(namespace, id).await?;
        if family.is_empty() {
            return Ok(0);
        }

        let mut deleted = 0usize;
        let mut ids = Vec::with_capacity(family.len());

        for document in family {
            deleted += self
                .storage
                .delete_document(namespace, &document.id)
                .await?
                .min(1);
            ids.push(document.id);
        }

        if let Some(bm25_writer) = &self.bm25_writer
            && !ids.is_empty()
        {
            bm25_writer.delete_documents(&ids).await?;
        }

        Ok(deleted)
    }

    fn preferred_memory_family_document(
        mut family: Vec<ChromaDocument>,
        requested_id: &str,
    ) -> Option<ChromaDocument> {
        fn rank(layer: Option<SliceLayer>) -> u8 {
            match layer {
                None => 0,
                Some(SliceLayer::Outer) => 1,
                Some(SliceLayer::Middle) => 2,
                Some(SliceLayer::Inner) => 3,
                Some(SliceLayer::Core) => 4,
            }
        }

        fn chunk_index(document: &ChromaDocument) -> usize {
            document
                .metadata
                .get("chunk_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(usize::MAX)
        }

        family.sort_by_key(|document| {
            if document.id == requested_id {
                (0_u8, 0_u8, 0_usize)
            } else {
                (1_u8, rank(document.slice_layer()), chunk_index(document))
            }
        });

        family.into_iter().next()
    }

    /// Get which MLX server we're connected to (for health/status reporting)
    pub fn mlx_connected_to(&self) -> String {
        // This is safe because mlx_bridge is required and always initialized
        if let Ok(bridge) = self.mlx_bridge.try_lock() {
            bridge.connected_to().to_string()
        } else {
            "mlx (lock held)".to_string()
        }
    }

    pub async fn index_document(&self, path: &Path, namespace: Option<&str>) -> Result<()> {
        self.index_document_with_mode(path, namespace, SliceMode::default())
            .await
    }

    /// Index a document with explicit slice mode
    pub async fn index_document_with_mode(
        &self,
        path: &Path,
        namespace: Option<&str>,
        slice_mode: SliceMode,
    ) -> Result<()> {
        self.index_document_internal(path, namespace, None, slice_mode)
            .await
    }

    /// Index a document with optional preprocessing to filter noise
    pub async fn index_document_with_preprocessing(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: PreprocessingConfig,
    ) -> Result<()> {
        self.index_document_internal(path, namespace, Some(preprocess_config), SliceMode::Flat)
            .await
    }

    /// Index a document with deduplication (skips if exact content already exists)
    ///
    /// For JSON files with arrays (conversations, sessions), automatically splits
    /// into multiple documents when using Onion or OnionFast slice modes.
    pub async fn index_document_with_dedup(
        &self,
        path: &Path,
        namespace: Option<&str>,
        slice_mode: SliceMode,
    ) -> Result<IndexResult> {
        // Security: validate path before any file operations
        let validated_path = crate::path_utils::validate_read_path(path)?;
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);

        // For JSON files, ALWAYS use JSON-aware extraction (smart conversation detection)
        // This allows conversations to be extracted as individual messages regardless of slice_mode
        let is_json = validated_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false);

        if is_json || matches!(slice_mode, SliceMode::Onion | SliceMode::OnionFast) {
            return self
                .index_document_with_json_awareness(&validated_path, ns, slice_mode)
                .await;
        }

        // For non-JSON Flat mode, use existing behavior (single document)
        let text = self.extract_text(&validated_path).await?;

        // Compute content hash BEFORE any processing
        let content_hash = compute_content_hash(&text);

        // Check if this exact content already exists
        if self.storage.has_content_hash(ns, &content_hash).await? {
            debug!(
                "Skipping duplicate content: {} (hash: {})",
                path.display(),
                &content_hash[..16]
            );
            return Ok(IndexResult::Skipped {
                reason: "exact duplicate".to_string(),
                content_hash,
            });
        }

        let base_metadata = json!({
            "path": path.to_str(),
            "slice_mode": "flat",
            "content_hash": &content_hash,
        });

        let (chunks_indexed, embedder_ms, tokens_estimated) = self
            .index_with_flat_chunking_and_hash(&text, ns, path, base_metadata, &content_hash)
            .await?;

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
            embedder_ms: Some(embedder_ms),
            tokens_estimated: Some(tokens_estimated),
        })
    }

    /// Index a document with JSON-awareness: for JSON arrays, each element
    /// becomes a separate onion-sliced document.
    ///
    /// This is critical for conversation/session files where a single JSON file
    /// may contain hundreds of messages that should each have their own onion slices.
    async fn index_document_with_json_awareness(
        &self,
        path: &Path,
        namespace: &str,
        slice_mode: SliceMode,
    ) -> Result<IndexResult> {
        // Extract documents (may be multiple for JSON arrays)
        let documents = self.extract_json_documents(path).await?;

        let mut total_chunks = 0;
        let mut total_embedder_ms = 0u64;
        let mut total_tokens_estimated = 0usize;
        let mut saw_metrics = false;
        let mut skipped_docs = 0;
        let file_content_hash = match crate::path_utils::safe_read_to_string_async(path).await {
            Ok((_p, content)) => compute_content_hash(&content),
            Err(_) => compute_content_hash(""),
        };

        for (doc_id, content, mut doc_metadata) in documents {
            if content.len() < 50 {
                continue; // Skip very small documents
            }

            // Compute per-document hash for dedup
            let doc_hash = compute_content_hash(&content);

            // Check if this document already exists
            if self.storage.has_content_hash(namespace, &doc_hash).await? {
                skipped_docs += 1;
                continue;
            }

            // Merge file-level metadata into document metadata
            if let serde_json::Value::Object(ref mut map) = doc_metadata {
                map.insert("doc_id".to_string(), json!(doc_id));
                map.insert("content_hash".to_string(), json!(doc_hash));
                map.insert("file_hash".to_string(), json!(&file_content_hash));
                map.insert(
                    "slice_mode".to_string(),
                    json!(match slice_mode {
                        SliceMode::Onion => "onion",
                        SliceMode::OnionFast => "onion-fast",
                        SliceMode::Flat => "flat",
                    }),
                );
            }

            let (chunks, embedder_ms, tokens_estimated) = match slice_mode {
                SliceMode::Onion => {
                    let (chunks, embedder_ms, tokens_estimated) = self
                        .index_with_onion_slicing_and_hash(
                            &content,
                            namespace,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?;
                    (chunks, Some(embedder_ms), Some(tokens_estimated))
                }
                SliceMode::OnionFast => {
                    let (chunks, embedder_ms, tokens_estimated) = self
                        .index_with_onion_slicing_fast_and_hash(
                            &content,
                            namespace,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?;
                    (chunks, Some(embedder_ms), Some(tokens_estimated))
                }
                SliceMode::Flat => {
                    let (chunks, embedder_ms, tokens_estimated) = self
                        .index_with_flat_chunking_and_hash(
                            &content,
                            namespace,
                            path,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?;
                    (chunks, Some(embedder_ms), Some(tokens_estimated))
                }
            };

            total_chunks += chunks;
            if let Some(embedder_ms) = embedder_ms {
                total_embedder_ms += embedder_ms;
                saw_metrics = true;
            }
            if let Some(tokens_estimated) = tokens_estimated {
                total_tokens_estimated += tokens_estimated;
                saw_metrics = true;
            }
        }

        if total_chunks == 0 && skipped_docs > 0 {
            return Ok(IndexResult::Skipped {
                reason: format!("all {} documents already indexed", skipped_docs),
                content_hash: file_content_hash,
            });
        }

        tracing::info!(
            "JSON-aware indexing: {} -> {} chunks ({} docs skipped)",
            path.display(),
            total_chunks,
            skipped_docs
        );

        Ok(IndexResult::Indexed {
            chunks_indexed: total_chunks,
            content_hash: file_content_hash,
            embedder_ms: saw_metrics.then_some(total_embedder_ms),
            tokens_estimated: saw_metrics.then_some(total_tokens_estimated),
        })
    }

    /// Index a document with preprocessing and deduplication
    pub async fn index_document_with_preprocessing_and_dedup(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: PreprocessingConfig,
    ) -> Result<IndexResult> {
        let text = self.extract_text(path).await?;
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);

        // Compute content hash BEFORE preprocessing (hash original content)
        let content_hash = compute_content_hash(&text);

        // Check if this exact content already exists
        if self.storage.has_content_hash(ns, &content_hash).await? {
            debug!(
                "Skipping duplicate content: {} (hash: {})",
                path.display(),
                &content_hash[..16]
            );
            return Ok(IndexResult::Skipped {
                reason: "exact duplicate".to_string(),
                content_hash,
            });
        }

        // Now preprocess for indexing
        let preprocessor = Preprocessor::new(preprocess_config);
        let cleaned = preprocessor.extract_semantic_content(&text);
        tracing::info!(
            "Preprocessing: {} chars -> {} chars ({:.1}% reduction)",
            text.len(),
            cleaned.len(),
            (1.0 - (cleaned.len() as f32 / text.len() as f32)) * 100.0
        );

        let base_metadata = json!({
            "path": path.to_str(),
            "slice_mode": "flat",
            "content_hash": &content_hash,
        });

        let (chunks_indexed, embedder_ms, tokens_estimated) = self
            .index_with_flat_chunking_and_hash(&cleaned, ns, path, base_metadata, &content_hash)
            .await?;

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
            embedder_ms: Some(embedder_ms),
            tokens_estimated: Some(tokens_estimated),
        })
    }

    async fn index_document_internal(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: Option<PreprocessingConfig>,
        slice_mode: SliceMode,
    ) -> Result<()> {
        // Security: validate path before any file operations
        let validated_path = crate::path_utils::validate_read_path(path)?;
        let text = self.extract_text(&validated_path).await?;

        // Optionally preprocess the text to remove noise
        let text = if let Some(config) = preprocess_config {
            let preprocessor = Preprocessor::new(config);
            let cleaned = preprocessor.extract_semantic_content(&text);
            tracing::info!(
                "Preprocessing: {} chars -> {} chars ({:.1}% reduction)",
                text.len(),
                cleaned.len(),
                (1.0 - (cleaned.len() as f32 / text.len() as f32)) * 100.0
            );
            cleaned
        } else {
            text
        };

        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let base_metadata = json!({
            "path": validated_path.to_str(),
            "slice_mode": match slice_mode {
                SliceMode::Onion => "onion",
                SliceMode::OnionFast => "onion-fast",
                SliceMode::Flat => "flat",
            }
        });

        match slice_mode {
            SliceMode::Onion => {
                self.index_with_onion_slicing(&text, ns, base_metadata)
                    .await
            }
            SliceMode::OnionFast => {
                self.index_with_onion_slicing_fast(&text, ns, base_metadata)
                    .await
            }
            SliceMode::Flat => {
                self.index_with_flat_chunking(&text, ns, path, base_metadata)
                    .await
            }
        }
    }

    /// Index using onion slice architecture (hierarchical embeddings)
    async fn index_with_onion_slicing(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
    ) -> Result<()> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices(text, &base_metadata, &config);
        let total_slices = slices.len();

        tracing::info!(
            "Onion slicing: {} chars -> {} slices (outer/middle/inner/core)",
            text.len(),
            total_slices
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embeddings = self.embed_chunks(&batch_contents).await?;

            // Create documents from this batch
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                let doc = ChromaDocument::from_onion_slice(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                );
                batch_docs.push(doc);
            }

            // Flush this batch to storage
            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(())
    }

    /// Fast onion slicing (outer + core only, no hash)
    async fn index_with_onion_slicing_fast(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
    ) -> Result<()> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices_fast(text, &base_metadata, &config);
        let total_slices = slices.len();

        tracing::info!(
            "Fast onion slicing: {} chars -> {} slices (outer/core only)",
            text.len(),
            total_slices
        );

        let mut total_stored = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embeddings = self.embed_chunks(&batch_contents).await?;

            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                let doc = ChromaDocument::from_onion_slice(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                );
                batch_docs.push(doc);
            }

            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(())
    }

    /// Index using onion slice architecture with content hash for deduplication
    async fn index_with_onion_slicing_and_hash(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<(usize, u64, usize)> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices(text, &base_metadata, &config);
        let total_slices = slices.len();
        let mut tokens_estimated = 0;
        let token_config = crate::embeddings::TokenConfig::default();

        tracing::info!(
            "Onion slicing: {} chars -> {} slices (outer/middle/inner/core)",
            text.len(),
            total_slices
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        let mut total_embedder_ms = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            // Estimate tokens for this batch
            for slice in batch {
                tokens_estimated +=
                    crate::embeddings::estimate_tokens(&slice.content, &token_config);
            }

            // Embed this batch
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embed_started_at = std::time::Instant::now();
            let embeddings = self.embed_chunks(&batch_contents).await?;
            total_embedder_ms += embed_started_at.elapsed().as_millis() as u64;

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                // Dual hash: file_hash for provenance, content_hash for per-slice dedup
                let slice_hash = compute_content_hash(&slice.content);
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("file_hash".to_string(), json!(content_hash));
                }
                let doc = ChromaDocument::from_onion_slice_with_hash(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    slice_hash,
                );
                batch_docs.push(doc);
            }

            // Flush this batch to storage
            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok((total_slices, total_embedder_ms, tokens_estimated))
    }

    /// Index using fast onion slice architecture (outer + core only)
    /// 2x faster than full onion, good for bulk indexing
    async fn index_with_onion_slicing_fast_and_hash(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<(usize, u64, usize)> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices_fast(text, &base_metadata, &config);
        let total_slices = slices.len();
        let mut tokens_estimated = 0;
        let token_config = crate::embeddings::TokenConfig::default();

        tracing::info!(
            "Fast onion slicing: {} chars -> {} slices (outer/core only)",
            text.len(),
            total_slices
        );

        // Process in batches
        let mut total_stored = 0;
        let mut total_embedder_ms = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            // Estimate tokens for this batch
            for slice in batch {
                tokens_estimated +=
                    crate::embeddings::estimate_tokens(&slice.content, &token_config);
            }

            // Embed this batch
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embed_started_at = std::time::Instant::now();
            let embeddings = self.embed_chunks(&batch_contents).await?;
            total_embedder_ms += embed_started_at.elapsed().as_millis() as u64;

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                // Dual hash: file_hash for provenance, content_hash for per-slice dedup
                let slice_hash = compute_content_hash(&slice.content);
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("file_hash".to_string(), json!(content_hash));
                }
                let doc = ChromaDocument::from_onion_slice_with_hash(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    slice_hash,
                );
                batch_docs.push(doc);
            }

            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok((total_slices, total_embedder_ms, tokens_estimated))
    }

    /// Index using traditional flat chunking (backward compatible)
    async fn index_with_flat_chunking(
        &self,
        text: &str,
        namespace: &str,
        path: &Path,
        base_metadata: serde_json::Value,
    ) -> Result<()> {
        // Chunk the text
        let chunks = self.chunk_text(text, 512, 128)?;
        let total_chunks = chunks.len();

        tracing::info!(
            "Flat chunking: {} chars -> {} chunks",
            text.len(),
            total_chunks
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        let mut global_idx = 0;
        for batch in chunks.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let embeddings = self.embed_chunks(batch).await?;

            // Create documents from this batch
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                }

                let doc = ChromaDocument::new_flat(
                    format!("{}_{}", path.to_str().unwrap_or("unknown"), global_idx),
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            // Flush this batch to storage
            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
        }

        Ok(())
    }

    /// Index using traditional flat chunking with content hash for deduplication
    async fn index_with_flat_chunking_and_hash(
        &self,
        text: &str,
        namespace: &str,
        path: &Path,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<(usize, u64, usize)> {
        // Chunk the text
        let chunks = self.chunk_text(text, 512, 128)?;
        let total_chunks = chunks.len();
        let mut tokens_estimated = 0;
        let token_config = crate::embeddings::TokenConfig::default();

        tracing::info!(
            "Flat chunking: {} chars -> {} chunks",
            text.len(),
            total_chunks
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        let mut global_idx = 0;
        let mut total_embedder_ms = 0;
        for batch in chunks.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            tokens_estimated += batch
                .iter()
                .map(|chunk| crate::embeddings::estimate_tokens(chunk, &token_config))
                .sum::<usize>();
            let embed_started_at = std::time::Instant::now();
            let embeddings = self.embed_chunks(batch).await?;
            total_embedder_ms += embed_started_at.elapsed().as_millis() as u64;

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                }

                // Dual hash: file_hash for provenance, content_hash for per-chunk dedup
                let chunk_hash = compute_content_hash(chunk);
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("file_hash".to_string(), json!(content_hash));
                }
                let doc = ChromaDocument::new_flat_with_hash(
                    format!("{}_{}", path.to_str().unwrap_or("unknown"), global_idx),
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                    chunk_hash,
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            // Flush this batch to storage
            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
        }

        Ok((total_chunks, total_embedder_ms, tokens_estimated))
    }

    async fn index_flat_memory_family_with_hash(
        &self,
        text: &str,
        namespace: &str,
        original_id: &str,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
        let chunks = self.chunk_text(text, 512, 128)?;
        let total_chunks = chunks.len();

        tracing::info!(
            "Flat memory chunking: {} chars -> {} chunks",
            text.len(),
            total_chunks
        );

        let mut total_stored = 0;
        let mut global_idx = 0;
        for batch in chunks.chunks(STORAGE_BATCH_SIZE) {
            let embeddings = self.embed_chunks(batch).await?;

            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                    map.insert("file_hash".to_string(), json!(content_hash));
                    map.insert("original_id".to_string(), json!(original_id));
                }

                let doc_id = if total_chunks == 1 {
                    original_id.to_string()
                } else {
                    format!("{original_id}::chunk::{global_idx}")
                };

                let chunk_hash = compute_content_hash(chunk);
                let doc = ChromaDocument::new_flat_with_hash(
                    doc_id,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                    chunk_hash,
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            self.persist_documents(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!(
                "Stored {}/{} flat memory chunks for {}",
                total_stored,
                total_chunks,
                original_id
            );
        }

        Ok(total_chunks)
    }

    pub async fn index_text(
        &self,
        namespace: Option<&str>,
        id: String,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<String> {
        self.index_text_with_mode(namespace, id, text, metadata, SliceMode::default())
            .await
    }

    /// Index text with explicit slice mode
    pub async fn index_text_with_mode(
        &self,
        namespace: Option<&str>,
        id: String,
        text: String,
        metadata: serde_json::Value,
        slice_mode: SliceMode,
    ) -> Result<String> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE).to_string();
        let slice_mode_name = match slice_mode {
            SliceMode::Onion => "onion",
            SliceMode::OnionFast => "onion-fast",
            SliceMode::Flat => "flat",
        };

        match slice_mode {
            SliceMode::Onion | SliceMode::OnionFast => {
                // For onion modes, ignore the provided ID and use generated slice IDs
                let config = OnionSliceConfig::default();
                let slices = if slice_mode == SliceMode::OnionFast {
                    create_onion_slices_fast(&text, &metadata, &config)
                } else {
                    create_onion_slices(&text, &metadata, &config)
                };

                let slice_contents: Vec<String> =
                    slices.iter().map(|s| s.content.clone()).collect();
                let embeddings = self.embed_chunks(&slice_contents).await?;

                let mut documents = Vec::with_capacity(slices.len());
                for (slice, embedding) in slices.iter().zip(embeddings.iter()) {
                    let mut meta = metadata.clone();
                    if let serde_json::Value::Object(ref mut map) = meta {
                        map.insert("layer".to_string(), json!(slice.layer.name()));
                        map.insert("original_id".to_string(), json!(id));
                        map.insert("slice_mode".to_string(), json!(slice_mode_name));
                    }

                    let doc = ChromaDocument::from_onion_slice(
                        slice,
                        ns.clone(),
                        embedding.clone(),
                        meta,
                    );
                    documents.push(doc);
                }

                self.persist_documents(documents).await?;

                // Return the outer slice ID (what search will hit first)
                Ok(slices
                    .iter()
                    .find(|s| s.layer == SliceLayer::Outer)
                    .map(|s| s.id.clone())
                    .unwrap_or(id))
            }
            SliceMode::Flat => {
                let embedding = self.embed_query(&text).await?;
                let mut metadata = metadata;
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("slice_mode".to_string(), json!(slice_mode_name));
                }
                let doc = ChromaDocument::new_flat(id.clone(), ns, embedding, metadata, text);
                self.persist_documents(vec![doc]).await?;
                Ok(id)
            }
        }
    }

    async fn index_text_memory_family_with_hash(
        &self,
        namespace: &str,
        id: &str,
        text: &str,
        metadata: serde_json::Value,
        slice_mode: SliceMode,
    ) -> Result<()> {
        let slice_mode_name = match slice_mode {
            SliceMode::Onion => "onion",
            SliceMode::OnionFast => "onion-fast",
            SliceMode::Flat => "flat",
        };
        let content_hash = compute_content_hash(text);
        let mut metadata = metadata;

        if let serde_json::Value::Object(ref mut map) = metadata {
            map.insert("slice_mode".to_string(), json!(slice_mode_name));
            if matches!(slice_mode, SliceMode::Onion | SliceMode::OnionFast) {
                map.insert("original_id".to_string(), json!(id));
            }
        }

        match slice_mode {
            SliceMode::Onion => {
                self.index_with_onion_slicing_and_hash(text, namespace, metadata, &content_hash)
                    .await?;
            }
            SliceMode::OnionFast => {
                self.index_with_onion_slicing_fast_and_hash(
                    text,
                    namespace,
                    metadata,
                    &content_hash,
                )
                .await?;
            }
            SliceMode::Flat => {
                self.index_flat_memory_family_with_hash(
                    text,
                    namespace,
                    id,
                    metadata,
                    &content_hash,
                )
                .await?;
            }
        }

        Ok(())
    }

    pub async fn memory_upsert(
        &self,
        namespace: &str,
        id: String,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<()> {
        let slice_mode = match metadata
            .get("slice_mode")
            .and_then(|value| value.as_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("onion") => SliceMode::Onion,
            Some("onion-fast") | Some("onion_fast") | Some("fast") => SliceMode::OnionFast,
            Some("flat") | None => SliceMode::Flat,
            Some(other) => {
                return Err(anyhow!(
                    "Unsupported metadata.slice_mode '{}'. Use 'flat', 'onion', or 'onion-fast'.",
                    other
                ));
            }
        };

        self.delete_memory_family(namespace, &id).await?;
        self.index_text_memory_family_with_hash(namespace, &id, &text, metadata, slice_mode)
            .await?;
        Ok(())
    }

    pub async fn lookup_memory(&self, namespace: &str, id: &str) -> Result<Option<SearchResult>> {
        if let Some(doc) = self.storage.get_document(namespace, id).await? {
            let layer = doc.slice_layer();
            return Ok(Some(SearchResult {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document,
                score: 1.0,
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            }));
        }

        if let Some(doc) = Self::preferred_memory_family_document(
            self.load_memory_family(namespace, id).await?,
            id,
        ) {
            let layer = doc.slice_layer();
            return Ok(Some(SearchResult {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document,
                score: 1.0,
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            }));
        }

        Ok(None)
    }

    pub async fn remove_memory(&self, namespace: &str, id: &str) -> Result<usize> {
        self.delete_memory_family(namespace, id).await
    }

    pub async fn clear_namespace(&self, namespace: &str) -> Result<usize> {
        self.clear_namespace_from_indices(namespace).await
    }

    pub async fn search_memory(
        &self,
        namespace: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(Some(namespace), query, k, SearchOptions::default())
            .await
    }

    /// Search with layer filter - returns only outer slices by default (efficient context usage)
    pub async fn memory_search_with_layer(
        &self,
        namespace: &str,
        query: &str,
        k: usize,
        layer: Option<SliceLayer>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(
            Some(namespace),
            query,
            k,
            SearchOptions {
                layer_filter: layer,
                project_filter: None,
            },
        )
        .await
    }

    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.search_inner(None, query, k).await
    }

    /// Legacy search method for backward compatibility
    pub async fn search_inner(
        &self,
        namespace: Option<&str>,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(namespace, query, k, SearchOptions::default())
            .await
    }

    /// Search with full options including layer filtering
    pub async fn search_with_options(
        &self,
        namespace: Option<&str>,
        query: &str,
        k: usize,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed_query(query).await?;
        let candidate_multiplier = if options.project_filter.is_some() {
            8
        } else {
            3
        };

        let mut candidates = self
            .storage
            .search_store_with_layer(
                namespace,
                query_embedding.clone(),
                k * candidate_multiplier,
                options.layer_filter,
            )
            .await?;

        if let Some(project) = options.project_filter.as_deref() {
            candidates.retain(|candidate| metadata_matches_project(&candidate.metadata, project));
        }

        // Rerank if we have candidates
        if !candidates.is_empty() {
            let documents: Vec<String> = candidates.iter().map(|c| c.document.clone()).collect();
            let metadatas: Vec<serde_json::Value> =
                candidates.iter().map(|c| c.metadata.clone()).collect();

            // Try MLX reranker; fallback to cosine if rerank fails
            let reranked = match self.mlx_bridge.lock().await.rerank(query, &documents).await {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!("MLX rerank failed, using cosine fallback: {}", e);
                    None
                }
            };

            let reranked = if let Some(r) = reranked {
                r
            } else {
                // Cosine fallback
                let doc_embeddings = self.ensure_doc_embeddings(&documents, &candidates).await?;
                let scores = doc_embeddings
                    .iter()
                    .enumerate()
                    .map(|(idx, emb)| (idx, cosine(&query_embedding, emb)))
                    .collect::<Vec<_>>();
                let mut scores = scores;
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores
            };

            // Return top-k reranked results with onion slice info
            let results: Vec<SearchResult> = reranked
                .into_iter()
                .take(k)
                .filter_map(|(idx, score)| {
                    candidates.get(idx).map(|candidate| {
                        SearchResult {
                            id: candidate.id.clone(),
                            namespace: candidate.namespace.clone(),
                            text: candidate.document.clone(),
                            score,
                            metadata: metadatas.get(idx).cloned().unwrap_or_else(|| json!({})),
                            // Onion slice fields
                            layer: candidate.slice_layer(),
                            parent_id: candidate.parent_id.clone(),
                            children_ids: candidate.children_ids.clone(),
                            keywords: candidate.keywords.clone(),
                        }
                    })
                })
                .collect();

            return Ok(results);
        }

        Ok(vec![])
    }

    /// Expand a search result to get its children (drill down in onion hierarchy)
    pub async fn expand_result(&self, namespace: &str, id: &str) -> Result<Vec<SearchResult>> {
        let children = self.storage.get_children(namespace, id).await?;
        Ok(children
            .into_iter()
            .map(|doc| {
                let layer = doc.slice_layer();
                SearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    text: doc.document,
                    score: 1.0,
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                }
            })
            .collect())
    }

    /// Get the parent of a search result (drill up in onion hierarchy)
    pub async fn get_parent_result(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<SearchResult>> {
        if let Some(parent) = self.storage.get_parent(namespace, id).await? {
            let layer = parent.slice_layer();
            return Ok(Some(SearchResult {
                id: parent.id,
                namespace: parent.namespace,
                text: parent.document,
                score: 1.0,
                metadata: parent.metadata,
                layer,
                parent_id: parent.parent_id,
                children_ids: parent.children_ids,
                keywords: parent.keywords,
            }));
        }
        Ok(None)
    }

    async fn extract_text(&self, path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "pdf" {
            // pdf_extract is blocking; offload to blocking thread
            let path = path.to_path_buf();
            let pdf_text =
                tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path)).await??;
            return Ok(pdf_text);
        }

        // Default: treat as UTF-8 text (validated read)
        let (_p, content) = crate::path_utils::safe_read_to_string_async(path).await?;
        Ok(content)
    }

    /// Extract multiple documents from a JSON file if it contains an array.
    /// For non-array JSON or other file types, returns a single document.
    ///
    /// This enables proper onion slicing for conversation/session files where
    /// dialog-like content should be normalized into turn documents before slicing.
    ///
    /// Returns: Vec of (doc_id, content, metadata) tuples
    async fn extract_json_documents(
        &self,
        path: &Path,
    ) -> Result<Vec<(String, String, serde_json::Value)>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if matches!(ext.as_str(), "md" | "markdown") {
            let (_p, raw) = crate::path_utils::safe_read_to_string_async(path).await?;
            if let Some(docs) = extract_markdown_transcript_documents(&raw, path) {
                tracing::info!(
                    "Markdown transcript detected: {} -> {} turn documents",
                    path.display(),
                    docs.len()
                );
                return Ok(docs);
            }

            let doc_id = format!("{}:0", path.display());
            let metadata = json!({ "path": path.to_str(), "index": 0 });
            return Ok(vec![(doc_id, raw, metadata)]);
        }

        // Only process JSON files specially
        if ext != "json" {
            let text = self.extract_text(path).await?;
            let doc_id = format!("{}:0", path.display());
            let metadata = json!({ "path": path.to_str(), "index": 0 });
            return Ok(vec![(doc_id, text, metadata)]);
        }

        // Try to parse as JSON (validated read)
        let (_p, raw) = crate::path_utils::safe_read_to_string_async(path).await?;
        let parsed: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => {
                // Not valid JSON, treat as text
                let doc_id = format!("{}:0", path.display());
                let metadata = json!({ "path": path.to_str(), "index": 0 });
                return Ok(vec![(doc_id, raw, metadata)]);
            }
        };

        // Check if it's an array
        if let serde_json::Value::Array(arr) = parsed {
            let mut docs = Vec::new();
            let mut used_smart_extraction = false;

            // Try smart conversation extraction for each array element
            for item in arr.iter() {
                if let Some(mut conv_docs) = extract_conversation_documents(item, path) {
                    docs.append(&mut conv_docs);
                    used_smart_extraction = true;
                }
            }

            // If smart extraction found conversations, use those
            if used_smart_extraction && !docs.is_empty() {
                tracing::info!(
                    "Conversation array detected: {} -> {} turn documents",
                    path.display(),
                    docs.len()
                );
                return Ok(docs);
            }

            // Fallback to element-by-element extraction
            docs.clear();
            for (idx, item) in arr.iter().enumerate() {
                let doc_id = format!("{}:{}", path.display(), idx);
                let content = extract_json_element_content(item);
                if content.len() > 50 {
                    // Skip very small elements
                    let metadata = json!({
                        "path": path.to_str(),
                        "index": idx,
                        "total_elements": arr.len(),
                        "element_type": detect_json_element_type(item),
                    });
                    docs.push((doc_id, content, metadata));
                }
            }
            if docs.is_empty() {
                // Fallback if all elements were too small
                let doc_id = format!("{}:0", path.display());
                let metadata = json!({ "path": path.to_str(), "index": 0 });
                return Ok(vec![(doc_id, raw, metadata)]);
            }
            tracing::info!(
                "JSON array detected: {} -> {} documents",
                path.display(),
                docs.len()
            );
            return Ok(docs);
        }

        // Try smart conversation extraction first
        if let Some(docs) = extract_conversation_documents(&parsed, path) {
            return Ok(docs);
        }

        // Fallback: treat as single document
        let content = extract_json_element_content(&parsed);
        let doc_id = format!("{}:0", path.display());
        let metadata = json!({ "path": path.to_str(), "index": 0 });
        Ok(vec![(doc_id, content, metadata)])
    }

    async fn embed_chunks(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        // Use MLX for all embeddings (no FastEmbed fallback!)
        self.mlx_bridge.lock().await.embed_batch(chunks).await
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.mlx_bridge.lock().await.embed(query).await
    }

    async fn ensure_doc_embeddings(
        &self,
        documents: &[String],
        candidates: &[ChromaDocument],
    ) -> Result<Vec<Vec<f32>>> {
        // If storage returned embeddings, use them; otherwise embed via MLX
        let has_all = candidates.iter().all(|c| !c.embedding.is_empty());
        if has_all {
            return Ok(candidates.iter().map(|c| c.embedding.clone()).collect());
        }

        self.mlx_bridge.lock().await.embed_batch(documents).await
    }

    /// Sentence-aware chunking that respects semantic boundaries.
    ///
    /// Instead of cutting at fixed character positions, this method:
    /// 1. Splits text into sentences
    /// 2. Aggregates sentences until reaching target_size
    /// 3. Adds overlap by including the last 1-2 sentences from the previous chunk
    fn chunk_text(&self, text: &str, target_size: usize, overlap: usize) -> Result<Vec<String>> {
        let sentences = split_into_sentences(text);

        if sentences.is_empty() {
            return Ok(vec![text.to_string()]);
        }

        // For very short text, return as single chunk
        if text.chars().count() <= target_size {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut overlap_sentences: Vec<String> = Vec::new();

        // Target overlap in sentences (typically 1-2 sentences)
        let overlap_sentence_count = (overlap / 50).clamp(1, 3);

        for sentence in &sentences {
            let sentence_len = sentence.chars().count();
            let current_len = current_chunk.chars().count();

            // If adding this sentence exceeds max_size (target_size * 1.5), flush chunk
            let max_size = target_size + target_size / 2;
            if current_len + sentence_len > max_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());

                // Start new chunk with overlap from previous chunk
                current_chunk = overlap_sentences.join(" ");
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                overlap_sentences.clear();
            }

            current_chunk.push_str(sentence);
            current_chunk.push(' ');

            // Track last N sentences for overlap
            overlap_sentences.push(sentence.clone());
            if overlap_sentences.len() > overlap_sentence_count {
                overlap_sentences.remove(0);
            }

            // If chunk reached target size, flush it
            if current_chunk.chars().count() >= target_size {
                chunks.push(current_chunk.trim().to_string());

                // Start new chunk with overlap
                current_chunk = overlap_sentences.join(" ");
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                overlap_sentences.clear();
            }
        }

        // Don't forget the last chunk
        let remaining = current_chunk.trim();
        if !remaining.is_empty() {
            // If last chunk is very short, merge with previous if possible
            if remaining.chars().count() < target_size / 4 && !chunks.is_empty() {
                let last_idx = chunks.len() - 1;
                chunks[last_idx].push(' ');
                chunks[last_idx].push_str(remaining);
            } else {
                chunks.push(remaining.to_string());
            }
        }

        // Ensure we have at least one chunk
        if chunks.is_empty() {
            chunks.push(text.to_string());
        }

        Ok(chunks)
    }
}

// =============================================================================
// CONTEXT PREFIX INJECTION
// =============================================================================
//
// Each chunk contains document context for better semantic matching.
// This helps the embedding model understand "what this chunk is about"
// without needing to see the full document.
//
// Format: [Source: filename.ext] [Section: Header Name] \n\n <content>
// =============================================================================

/// Configuration for context prefix injection
#[derive(Debug, Clone)]
pub struct ContextPrefixConfig {
    /// Include source filename in prefix
    pub include_source: bool,
    /// Include section header in prefix (if detected)
    pub include_section: bool,
    /// Include document type hint
    pub include_doc_type: bool,
    /// Maximum prefix length (chars)
    pub max_prefix_length: usize,
}

impl Default for ContextPrefixConfig {
    fn default() -> Self {
        Self {
            include_source: true,
            include_section: true,
            include_doc_type: true,
            max_prefix_length: 100,
        }
    }
}

/// An enriched chunk with context prefix and metadata
#[derive(Debug, Clone)]
pub struct EnrichedChunk {
    /// Full content with context prefix prepended
    pub content: String,
    /// Original content without prefix (for display)
    pub original_content: String,
    /// Source document path
    pub doc_path: String,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Section header (if detected)
    pub section: Option<String>,
    /// Detected document type
    pub doc_type: Option<String>,
}

/// Create enriched chunks with context prefix injection
///
/// # Arguments
/// * `content` - The text content to chunk
/// * `doc_path` - Path to the source document
/// * `chunk_size` - Target chunk size in characters
/// * `overlap` - Overlap between chunks
/// * `config` - Context prefix configuration
///
/// # Returns
/// Vector of enriched chunks with context prefixes
pub fn create_enriched_chunks(
    content: &str,
    doc_path: &str,
    chunk_size: usize,
    overlap: usize,
    config: &ContextPrefixConfig,
) -> Vec<EnrichedChunk> {
    // Detect document type from extension
    let doc_type = detect_doc_type(doc_path);

    // Extract filename for source prefix
    let filename = std::path::Path::new(doc_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    // Split content into sections (based on headers)
    let sections = extract_sections(content);

    let mut enriched_chunks = Vec::new();
    let mut global_chunk_index = 0;

    for (section_header, section_content) in sections {
        // Chunk this section
        let chunks = smart_chunk_text(section_content, chunk_size, overlap);

        for chunk in chunks {
            // Build context prefix
            let prefix = build_context_prefix(
                filename,
                section_header.as_deref(),
                doc_type.as_deref(),
                config,
            );

            // Combine prefix with content
            let full_content = if prefix.is_empty() {
                chunk.clone()
            } else {
                format!("{}\n\n{}", prefix, chunk)
            };

            enriched_chunks.push(EnrichedChunk {
                content: full_content,
                original_content: chunk,
                doc_path: doc_path.to_string(),
                chunk_index: global_chunk_index,
                section: section_header.clone(),
                doc_type: doc_type.clone(),
            });

            global_chunk_index += 1;
        }
    }

    // If no chunks were created (e.g., empty content), create one
    if enriched_chunks.is_empty() && !content.trim().is_empty() {
        let prefix = build_context_prefix(filename, None, doc_type.as_deref(), config);
        let full_content = if prefix.is_empty() {
            content.to_string()
        } else {
            format!("{}\n\n{}", prefix, content)
        };

        enriched_chunks.push(EnrichedChunk {
            content: full_content,
            original_content: content.to_string(),
            doc_path: doc_path.to_string(),
            chunk_index: 0,
            section: None,
            doc_type,
        });
    }

    enriched_chunks
}

/// Build context prefix string
fn build_context_prefix(
    filename: &str,
    section: Option<&str>,
    doc_type: Option<&str>,
    config: &ContextPrefixConfig,
) -> String {
    let mut parts = Vec::new();

    if config.include_source && !filename.is_empty() {
        parts.push(format!("[Source: {}]", filename));
    }

    if config.include_section
        && let Some(sec) = section
    {
        parts.push(format!("[Section: {}]", sec));
    }

    if config.include_doc_type
        && let Some(dt) = doc_type
    {
        parts.push(format!("[Type: {}]", dt));
    }

    let prefix = parts.join(" ");

    // Truncate if too long
    if prefix.len() > config.max_prefix_length {
        prefix.chars().take(config.max_prefix_length).collect()
    } else {
        prefix
    }
}

/// Detect document type from file extension
fn detect_doc_type(path: &str) -> Option<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())?;

    let doc_type = match ext.as_str() {
        "rs" => "Rust source code",
        "py" => "Python source code",
        "js" | "jsx" => "JavaScript source code",
        "ts" | "tsx" => "TypeScript source code",
        "md" => "Markdown documentation",
        "txt" => "Plain text",
        "json" => "JSON data",
        "yaml" | "yml" => "YAML configuration",
        "toml" => "TOML configuration",
        "html" => "HTML document",
        "css" => "CSS stylesheet",
        "sql" => "SQL query",
        "sh" | "bash" => "Shell script",
        "pdf" => "PDF document",
        _ => return None,
    };

    Some(doc_type.to_string())
}

/// Extract sections from content based on markdown-style headers
fn extract_sections(content: &str) -> Vec<(Option<String>, &str)> {
    // Simple header detection for markdown-style headers
    let header_pattern = regex::Regex::new(r"(?m)^(#{1,6})\s+(.+)$").ok();

    if let Some(re) = header_pattern {
        let mut sections = Vec::new();
        let mut last_end = 0;
        let mut current_header: Option<String> = None;

        for caps in re.captures_iter(content) {
            let Some(full_match) = caps.get(0) else {
                continue;
            };
            let Some(header_match) = caps.get(2) else {
                continue;
            };
            let match_start = full_match.start();

            // Add previous section
            if match_start > last_end {
                let section_content = &content[last_end..match_start];
                if !section_content.trim().is_empty() {
                    sections.push((current_header.clone(), section_content.trim()));
                }
            }

            current_header = Some(header_match.as_str().to_string());
            last_end = full_match.end();
        }

        // Add final section
        if last_end < content.len() {
            let section_content = &content[last_end..];
            if !section_content.trim().is_empty() {
                sections.push((current_header, section_content.trim()));
            }
        }

        if sections.is_empty() {
            vec![(None, content)]
        } else {
            sections
        }
    } else {
        vec![(None, content)]
    }
}

/// Smart text chunking respecting sentence boundaries
fn smart_chunk_text(text: &str, target_size: usize, overlap: usize) -> Vec<String> {
    let sentences = split_into_sentences(text);

    if sentences.is_empty() || text.chars().count() <= target_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut overlap_sentences: Vec<String> = Vec::new();
    let overlap_sentence_count = (overlap / 50).clamp(1, 3);

    for sentence in &sentences {
        let sentence_len = sentence.chars().count();
        let current_len = current_chunk.chars().count();
        let max_size = target_size + target_size / 2;

        if current_len + sentence_len > max_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = overlap_sentences.join(" ");
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            overlap_sentences.clear();
        }

        current_chunk.push_str(sentence);
        current_chunk.push(' ');

        overlap_sentences.push(sentence.clone());
        if overlap_sentences.len() > overlap_sentence_count {
            overlap_sentences.remove(0);
        }

        if current_chunk.chars().count() >= target_size {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = overlap_sentences.join(" ");
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            overlap_sentences.clear();
        }
    }

    let remaining = current_chunk.trim();
    if !remaining.is_empty() {
        if remaining.chars().count() < target_size / 4 && !chunks.is_empty() {
            let last_idx = chunks.len() - 1;
            chunks[last_idx].push(' ');
            chunks[last_idx].push_str(remaining);
        } else {
            chunks.push(remaining.to_string());
        }
    }

    if chunks.is_empty() {
        chunks.push(text.to_string());
    }

    chunks
}

/// Split text into sentences using common sentence boundaries.
/// Returns Vec of sentences with punctuation preserved.
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current.push(c);

        // Check for sentence ending
        if matches!(c, '.' | '!' | '?') {
            // Look ahead - if followed by whitespace or newline, it's likely end of sentence
            if let Some(&next) = chars.peek() {
                if next.is_whitespace() {
                    // Skip common abbreviations
                    let trimmed = current.trim();
                    let is_abbreviation = trimmed.ends_with("Mr.")
                        || trimmed.ends_with("Mrs.")
                        || trimmed.ends_with("Dr.")
                        || trimmed.ends_with("Prof.")
                        || trimmed.ends_with("vs.")
                        || trimmed.ends_with("etc.")
                        || trimmed.ends_with("e.g.")
                        || trimmed.ends_with("i.e.")
                        // Single letter abbreviations like "A." or "B."
                        || (trimmed.len() >= 2 && trimmed.chars().rev().nth(1).map(|c| c.is_uppercase()).unwrap_or(false));

                    if !is_abbreviation {
                        sentences.push(current.trim().to_string());
                        current = String::new();
                        // Skip the whitespace
                        chars.next();
                    }
                }
            } else {
                // End of text
                sentences.push(current.trim().to_string());
                current = String::new();
            }
        } else if c == '\n' {
            // Double newline often indicates paragraph break
            if let Some(&next) = chars.peek()
                && next == '\n'
            {
                if !current.trim().is_empty() {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
                chars.next(); // skip second newline
            }
        }
    }

    // Don't forget remaining text
    let remaining = current.trim();
    if !remaining.is_empty() {
        sentences.push(remaining.to_string());
    }

    sentences
}

/// Options for search operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOptions {
    /// Filter by onion slice layer (None = all layers)
    pub layer_filter: Option<SliceLayer>,
    /// Optional project identifier from metadata (e.g. project / project_id)
    pub project_filter: Option<String>,
}

impl SearchOptions {
    /// Search only outer slices (default for onion mode - minimum context, maximum navigation)
    pub fn outer_only() -> Self {
        Self {
            layer_filter: Some(SliceLayer::Outer),
            project_filter: None,
        }
    }

    /// Deep search - include all layers including Core
    pub fn deep() -> Self {
        Self {
            layer_filter: None,
            project_filter: None,
        }
    }

    pub fn with_project(mut self, project: Option<String>) -> Self {
        self.project_filter = project.filter(|value| !value.trim().is_empty());
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::outer_only()
    }
}

fn metadata_matches_project(metadata: &Value, project: &str) -> bool {
    let needle = project.trim();
    if needle.is_empty() {
        return true;
    }

    metadata.as_object().is_some_and(|object| {
        ["project", "project_id", "source_project"]
            .iter()
            .filter_map(|key| object.get(*key))
            .filter_map(|value| value.as_str())
            .any(|value| value.eq_ignore_ascii_case(needle))
    })
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub namespace: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    /// Onion slice layer (None for legacy flat chunks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<SliceLayer>,
    /// Parent slice ID for drilling up in hierarchy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Children slice IDs for drilling down in hierarchy
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children_ids: Vec<String>,
    /// Keywords extracted from this slice
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
}

impl SearchResult {
    /// Create a legacy result without onion slice fields
    pub fn new_legacy(
        id: String,
        namespace: String,
        text: String,
        score: f32,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id,
            namespace,
            text,
            score,
            metadata,
            layer: None,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
        }
    }

    /// Check if this result can be expanded (has children)
    pub fn can_expand(&self) -> bool {
        !self.children_ids.is_empty()
    }

    /// Check if this result has a parent to drill up to
    pub fn can_drill_up(&self) -> bool {
        self.parent_id.is_some()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::{
        OnionSliceConfig, SearchOptions, SliceLayer, create_onion_slices, create_onion_slices_fast,
        extract_conversation_documents, extract_keywords, extract_markdown_transcript_documents,
        hash_content, metadata_matches_project,
    };
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn short_hash_uses_sha256_prefix_with_minimum_length() {
        let hash = hash_content("same content");
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash, hash_content("same content"));
    }

    #[test]
    fn keyword_extraction_splits_paths_and_filters_session_tokens() {
        let keywords = extract_keywords(
            "/Users/silver/Git/tools/TwinSweep session 2ff4de8b9a4e1234567890abcdef notes",
            10,
        );

        assert!(keywords.contains(&"users".to_string()));
        assert!(keywords.contains(&"twinsweep".to_string()));
        assert!(!keywords.iter().any(|keyword| keyword.contains("2ff4de8b")));
    }

    #[test]
    fn search_options_can_carry_project_filter() {
        let options = SearchOptions::deep().with_project(Some("Vista".to_string()));
        assert_eq!(options.layer_filter, None);
        assert_eq!(options.project_filter.as_deref(), Some("Vista"));
    }

    #[test]
    fn project_match_uses_metadata_fields() {
        assert!(metadata_matches_project(
            &json!({"project": "Vista"}),
            "vista"
        ));
        assert!(metadata_matches_project(
            &json!({"project_id": "VetCoders"}),
            "vetcoders"
        ));
        assert!(!metadata_matches_project(
            &json!({"project": "rmcp-memex"}),
            "vista"
        ));
        assert_eq!(
            SearchOptions::default().layer_filter,
            Some(SliceLayer::Outer)
        );
    }

    #[test]
    fn markdown_transcript_extraction_builds_role_aware_turn_docs() {
        let raw = r#"[project: VetCoders/vibecrafted | agent: codex | date: 2026-03-30]

[signals]
Results:
- AICX lookup działa
[/signals]

[09:14:00] assistant: Tak, i to właśnie jest sedno: `aicx-dragon` to żywy endpoint MCP.
[09:15:33] user: ziom ale ty sobie sam skonfigurowałeś ~/.codex/config.toml
[09:15:47] assistant: Sprawdzam teraz lokalny kontrakt konfiguracji MCP dla Codexa.
[09:15:55] reasoning: **Checking config contract**
[09:16:06] assistant: Składnia configu wygląda już poprawnie według samego Codexa.
"#;

        let docs = extract_markdown_transcript_documents(raw, Path::new("sample.md"))
            .expect("expected transcript docs");

        assert_eq!(docs.len(), 2);
        assert!(docs[0].1.contains("Assistant response:"));
        assert!(docs[1].1.contains("User request:"));
        assert!(docs[1].1.contains("Reasoning focus:"));
        assert_eq!(docs[1].2["format"], "markdown_transcript");
        assert_eq!(docs[1].2["type"], "transcript_turn");
        assert_eq!(docs[1].2["project"], "VetCoders/vibecrafted");
    }

    #[test]
    fn short_structured_transcript_turns_keep_outer_slice_in_full_and_fast_modes() {
        let metadata = json!({
            "type": "transcript_turn",
            "format": "markdown_transcript"
        });
        let config = OnionSliceConfig::default();
        let content = "User request:\nDodaj progress do pipeline.\n\nAssistant response:\nPodepnę licznik etapów.";

        let full_layers: Vec<SliceLayer> = create_onion_slices(content, &metadata, &config)
            .into_iter()
            .map(|slice| slice.layer)
            .collect();
        let fast_layers: Vec<SliceLayer> = create_onion_slices_fast(content, &metadata, &config)
            .into_iter()
            .map(|slice| slice.layer)
            .collect();

        assert_eq!(full_layers, vec![SliceLayer::Outer, SliceLayer::Core]);
        assert_eq!(fast_layers, vec![SliceLayer::Outer, SliceLayer::Core]);
    }

    #[test]
    fn structured_markdown_outer_becomes_semantic_card() {
        let metadata = json!({
            "type": "transcript_turn",
            "format": "markdown_transcript",
            "project": "VetCoders/rmcp-memex",
            "agent": "codex"
        });
        let config = OnionSliceConfig {
            outer_target: 220,
            ..OnionSliceConfig::default()
        };
        let content = "User request:\nMake outer retrieval useful for transcript search.\n\nAssistant response:\nDecision: build semantic cards instead of keyword prefixes.\nNext action: add JSON slicing coverage.\n\nReasoning focus:\nOuter-only is the default search path, so weak outer text hides short turns.";

        let slices = create_onion_slices(content, &metadata, &config);
        let outer = &slices[0].content;
        let middle = &slices[1].content;
        let inner = &slices[2].content;

        assert!(outer.contains("Request:"));
        assert!(outer.contains("Response:"));
        assert!(outer.contains("Decision:"));
        assert!(!outer.starts_with('['));
        assert!(middle.contains("Decision:"));
        assert!(middle.contains("Next:"));
        assert!(inner.contains("Assistant response:"));
        assert!(inner.contains("Entities:"));
    }

    #[test]
    fn json_conversation_docs_flow_through_structured_semantic_slices() {
        let conversation = json!({
            "project": "VetCoders/rmcp-memex",
            "sessions": [
                {
                    "info": {
                        "sessionId": "session-1234567890"
                    },
                    "messages": [
                        {
                            "role": "user",
                            "text": "Please replace the keyword-prefixed outer summary with something that reads like a semantic card for search.",
                            "timestamp": "2026-04-12T04:00:00Z"
                        },
                        {
                            "role": "assistant",
                            "text": "Decision: we will use semantic cards for outer retrieval. Next action: add JSON regression coverage and preserve the plain text fallback.",
                            "timestamp": "2026-04-12T04:01:00Z"
                        },
                        {
                            "role": "user",
                            "text": "Keep the generic plain text path as a safe fallback.",
                            "timestamp": "2026-04-12T04:02:00Z"
                        },
                        {
                            "role": "assistant",
                            "text": "Reasoning: default search prefers outer-only, so the semantic card needs to surface the exchange even for short turns.",
                            "timestamp": "2026-04-12T04:03:00Z"
                        }
                    ]
                }
            ]
        });

        let docs = extract_conversation_documents(&conversation, Path::new("conversation.json"))
            .expect("expected conversation docs");
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0].2.get("format").and_then(|value| value.as_str()),
            Some("sessions")
        );
        assert_eq!(
            docs[0].2.get("turn_index").and_then(|value| value.as_u64()),
            Some(0)
        );
        assert!(docs[0].1.contains("User request:"));
        assert!(docs[0].1.contains("Assistant response:"));

        let config = OnionSliceConfig {
            outer_target: 220,
            ..OnionSliceConfig::default()
        };
        let slices = create_onion_slices(&docs[0].1, &docs[0].2, &config);
        let outer = &slices[0].content;
        let middle = &slices[1].content;
        let inner = &slices[2].content;

        assert!(outer.contains("Request:"));
        assert!(outer.contains("Response:"));
        assert!(outer.contains("Decision:"));
        assert!(!outer.starts_with('['));
        assert!(middle.contains("Decision:"));
        assert!(middle.contains("Next:"));
        assert!(inner.contains("Assistant response:"));
    }

    #[test]
    fn plain_text_still_uses_generic_fallback_path() {
        let metadata = json!({
            "type": "note",
            "format": "markdown"
        });
        let config = OnionSliceConfig::default();
        let content = "The release workflow still needs a truthful browse surface. We should preserve the plain text fallback while improving structured conversation retrieval with semantic cards and regression tests so the generic path does not regress.";

        let slices = create_onion_slices(content, &metadata, &config);
        let outer = &slices[0].content;

        assert_eq!(
            slices.iter().map(|slice| slice.layer).collect::<Vec<_>>(),
            vec![
                SliceLayer::Outer,
                SliceLayer::Middle,
                SliceLayer::Inner,
                SliceLayer::Core
            ]
        );
        assert!(!outer.contains("Request:"));
        assert!(!outer.contains("Response:"));
    }
}
