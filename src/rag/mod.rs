use anyhow::Result;
use pdf_extract;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

use crate::{
    embeddings::MLXBridge,
    preprocessing::{PreprocessingConfig, Preprocessor},
    storage::{ChromaDocument, StorageManager},
};

// Async pipeline module for concurrent indexing
pub mod pipeline;
pub use pipeline::{
    Chunk, EmbeddedChunk, FileContent, PipelineConfig, PipelineResult, PipelineStats, run_pipeline,
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
    pub fn is_indexed(&self) -> bool {
        matches!(self, IndexResult::Indexed { .. })
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

/// Create onion slices from content
///
/// Algorithm:
/// 1. Full content -> CORE slice
/// 2. Extract key sentences -> INNER slice (~600 chars)
/// 3. Summarize to key points -> MIDDLE slice (~300 chars)
/// 4. Extract keywords/topic -> OUTER slice (~100 chars)
pub fn create_onion_slices(
    content: &str,
    _metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();

    // For very short content, just create a single Core slice
    if content.len() < config.min_content_for_slicing {
        let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
        let keywords = extract_keywords(content, 5);
        return vec![OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![],
            keywords,
        }];
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
    _metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();

    // For very short content, just create a single Core slice
    if content.len() < config.min_content_for_slicing {
        let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
        let keywords = extract_keywords(content, 5);
        return vec![OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![],
            keywords,
        }];
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
    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        if cleaned.len() >= 3 && !stop_set.contains(cleaned.as_str()) {
            *word_counts.entry(cleaned).or_insert(0) += 1;
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

/// Create short hash for document deduplication
fn hash_content(text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:x}", hasher.finish()).chars().take(12).collect()
}

/// Extract conversation documents from JSON with smart format detection.
/// Returns individual messages as separate documents with proper metadata.
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

            if let Some(serde_json::Value::Array(messages)) = session_obj.get("messages") {
                for (idx, msg) in messages.iter().enumerate() {
                    let msg_obj = match msg.as_object() {
                        Some(o) => o,
                        None => continue,
                    };

                    let role = msg_obj
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let text = msg_obj
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let timestamp = msg_obj
                        .get("timestamp")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    // Skip empty or too short messages
                    let text = text.trim();
                    if text.len() < 20 {
                        continue;
                    }

                    let content_hash = hash_content(text);
                    let doc_id = format!("msg-{}-{:04}-{}", session_short, idx, content_hash);

                    let metadata = json!({
                        "role": role,
                        "session": session_short,
                        "project": project,
                        "timestamp": timestamp,
                        "source": source_name,
                        "type": "conversation",
                        "format": "sessions"
                    });

                    docs.push((doc_id, text.to_string(), metadata));
                }
            }
        }

        if !docs.is_empty() {
            tracing::info!(
                "Sessions format detected: {} -> {} messages",
                source_path.display(),
                docs.len()
            );
            return Some(docs);
        }
    }

    // Pattern 2: [{uuid, name, messages: [{sender, text, created_at}]}]
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

        // Check if it looks like a conversation (messages with sender/role)
        let looks_like_conversation = messages.iter().any(|m| {
            m.get("sender").is_some()
                || m.get("role").is_some()
                || m.get("author").is_some()
        });

        if looks_like_conversation {
            let mut docs = Vec::new();
            for (idx, msg) in messages.iter().enumerate() {
                let msg_obj = match msg.as_object() {
                    Some(o) => o,
                    None => continue,
                };

                let role = msg_obj
                    .get("sender")
                    .or_else(|| msg_obj.get("role"))
                    .or_else(|| msg_obj.get("author").and_then(|a| a.get("role")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                // Normalize role names
                let role = match role {
                    "human" => "user",
                    "assistant" | "bot" => "assistant",
                    other => other,
                };

                // Extract text from various formats
                let text = msg_obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        // Handle content array (Claude format)
                        msg_obj.get("content").and_then(|c| {
                            if let Some(s) = c.as_str() {
                                Some(s)
                            } else if let Some(_arr) = c.as_array() {
                                // Collect text from content blocks
                                None // Handle below
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or("");

                // Handle content blocks array
                let text = if text.is_empty() {
                    if let Some(serde_json::Value::Array(content)) = msg_obj.get("content") {
                        content
                            .iter()
                            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join(" ")
                    } else {
                        String::new()
                    }
                } else {
                    text.to_string()
                };

                let timestamp = msg_obj
                    .get("created_at")
                    .or_else(|| msg_obj.get("timestamp"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let text = text.trim();
                if text.len() < 20 {
                    continue;
                }

                let content_hash = hash_content(text);
                let doc_id = format!("conv-{}-{:04}-{}", conv_short, idx, content_hash);

                let metadata = json!({
                    "role": role,
                    "conversation": conv_short,
                    "title": title,
                    "timestamp": timestamp,
                    "source": source_name,
                    "type": "conversation",
                    "format": "claude_web"
                });

                docs.push((doc_id, text.to_string(), metadata));
            }

            if !docs.is_empty() {
                tracing::info!(
                    "Conversation format detected: {} -> {} messages",
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
        let title = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut docs = Vec::new();
        for (idx, msg) in messages.iter().enumerate() {
            let msg_obj = match msg.as_object() {
                Some(o) => o,
                None => continue,
            };

            let role = msg_obj
                .get("sender")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            // Normalize role names
            let role = match role {
                "human" => "user",
                "assistant" | "bot" => "assistant",
                other => other,
            };

            let text = msg_obj
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let timestamp = msg_obj
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let text = text.trim();
            if text.len() < 20 {
                continue;
            }

            let content_hash = hash_content(text);
            let doc_id = format!("chat-{}-{:04}-{}", conv_short, idx, content_hash);

            let metadata = json!({
                "role": role,
                "conversation": conv_short,
                "title": title,
                "timestamp": timestamp,
                "source": source_name,
                "type": "conversation",
                "format": "claude_web"
            });

            docs.push((doc_id, text.to_string(), metadata));
        }

        if !docs.is_empty() {
            tracing::info!(
                "Claude.ai chat_messages format detected: {} -> {} messages",
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

        let mut docs = Vec::new();
        let mut entries: Vec<_> = mapping.iter().collect();
        // Try to sort by create_time if available
        entries.sort_by(|a, b| {
            let time_a = a.1.get("message")
                .and_then(|m| m.get("create_time"))
                .and_then(|t| t.as_f64())
                .unwrap_or(0.0);
            let time_b = b.1.get("message")
                .and_then(|m| m.get("create_time"))
                .and_then(|t| t.as_f64())
                .unwrap_or(0.0);
            time_a.partial_cmp(&time_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (idx, (_node_id, node)) in entries.iter().enumerate() {
            let message = match node.get("message") {
                Some(m) => m,
                None => continue,
            };

            let role = message
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            // Skip system messages
            if role == "system" {
                continue;
            }

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
                .unwrap_or_default();

            let timestamp = message
                .get("create_time")
                .and_then(|t| t.as_f64())
                .map(|ts| {
                    chrono::DateTime::from_timestamp(ts as i64, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default()
                })
                .unwrap_or_default();

            let text = text.trim();
            if text.len() < 20 {
                continue;
            }

            let content_hash = hash_content(text);
            let doc_id = format!("gpt-{}-{:04}-{}", conv_short, idx, content_hash);

            let metadata = json!({
                "role": role,
                "conversation": conv_short,
                "title": title,
                "timestamp": timestamp,
                "source": source_name,
                "type": "conversation",
                "format": "chatgpt"
            });

            docs.push((doc_id, text.to_string(), metadata));
        }

        if !docs.is_empty() {
            tracing::info!(
                "ChatGPT format detected: {} -> {} messages",
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
}

impl RAGPipeline {
    /// Create new RAGPipeline with MLXBridge (required, no fallback!)
    pub async fn new(
        mlx_bridge: Arc<Mutex<MLXBridge>>,
        storage: Arc<StorageManager>,
    ) -> Result<Self> {
        Ok(Self {
            mlx_bridge,
            storage,
        })
    }

    pub fn storage(&self) -> Arc<StorageManager> {
        self.storage.clone()
    }

    /// Refresh storage to see new data written by other processes
    pub async fn refresh(&self) -> Result<()> {
        self.storage.refresh().await
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

        let chunks_indexed = self
            .index_with_flat_chunking_and_hash(&text, ns, path, base_metadata, &content_hash)
            .await?;

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
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
        let mut skipped_docs = 0;
        // Path is validated by caller (index_document_with_dedup) via validate_read_path
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        let file_content_hash = compute_content_hash(
            &tokio::fs::read_to_string(path).await.unwrap_or_default(), // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        );

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

            // Check if this is a conversation message - store directly without chunking
            let is_conversation = doc_metadata
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t == "conversation")
                .unwrap_or(false);

            let chunks = if is_conversation {
                // Store conversation message as single document (no chunking!)
                self.index_conversation_message_direct(
                    &doc_id,
                    &content,
                    namespace,
                    doc_metadata,
                    &doc_hash,
                )
                .await?
            } else {
                // Index with onion/flat slicing for non-conversation content
                match slice_mode {
                    SliceMode::Onion => {
                        self.index_with_onion_slicing_and_hash(
                            &content,
                            namespace,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?
                    }
                    SliceMode::OnionFast => {
                        self.index_with_onion_slicing_fast_and_hash(
                            &content,
                            namespace,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?
                    }
                    SliceMode::Flat => {
                        self.index_with_flat_chunking_and_hash(
                            &content,
                            namespace,
                            path,
                            doc_metadata,
                            &doc_hash,
                        )
                        .await?
                    }
                }
            };

            total_chunks += chunks;
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

        let chunks_indexed = self
            .index_with_flat_chunking_and_hash(&cleaned, ns, path, base_metadata, &content_hash)
            .await?;

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
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
            self.storage.add_to_store(batch_docs).await?;
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

            self.storage.add_to_store(batch_docs).await?;
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
    ) -> Result<usize> {
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

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                let doc = ChromaDocument::from_onion_slice_with_hash(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    content_hash.to_string(),
                );
                batch_docs.push(doc);
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(total_slices)
    }

    /// Index using fast onion slice architecture (outer + core only)
    /// 2x faster than full onion, good for bulk indexing
    async fn index_with_onion_slicing_fast_and_hash(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices_fast(text, &base_metadata, &config);
        let total_slices = slices.len();

        tracing::info!(
            "Fast onion slicing: {} chars -> {} slices (outer/core only)",
            text.len(),
            total_slices
        );

        // Process in batches
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

                let doc = ChromaDocument::from_onion_slice_with_hash(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    content_hash.to_string(),
                );
                batch_docs.push(doc);
            }

            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(total_slices)
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
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
        }

        Ok(())
    }

    /// Index a conversation message directly without chunking.
    /// Each message is stored as a single document with its own embedding.
    /// Used for smart-extracted conversation messages where chunking would lose context.
    async fn index_conversation_message_direct(
        &self,
        doc_id: &str,
        text: &str,
        namespace: &str,
        metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
        // Embed the entire message as one unit
        let embedding = self.embed_query(text).await?;

        // Create document with the smart-extracted ID (msg-XXX-NNNN-HASH)
        let doc = ChromaDocument::new_flat_with_hash(
            doc_id.to_string(),
            namespace.to_string(),
            embedding,
            metadata,
            text.to_string(),
            content_hash.to_string(),
        );

        // Store directly
        self.storage.add_to_store(vec![doc]).await?;

        tracing::debug!(
            "Conversation message stored directly: {} ({} chars)",
            doc_id,
            text.len()
        );

        Ok(1) // One document stored
    }

    /// Index using traditional flat chunking with content hash for deduplication
    async fn index_with_flat_chunking_and_hash(
        &self,
        text: &str,
        namespace: &str,
        path: &Path,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
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

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                }

                let doc = ChromaDocument::new_flat_with_hash(
                    format!("{}_{}", path.to_str().unwrap_or("unknown"), global_idx),
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                    content_hash.to_string(),
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
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
                    }

                    let doc = ChromaDocument::from_onion_slice(
                        slice,
                        ns.clone(),
                        embedding.clone(),
                        meta,
                    );
                    documents.push(doc);
                }

                self.storage.add_to_store(documents).await?;

                // Return the outer slice ID (what search will hit first)
                Ok(slices
                    .iter()
                    .find(|s| s.layer == SliceLayer::Outer)
                    .map(|s| s.id.clone())
                    .unwrap_or(id))
            }
            SliceMode::Flat => {
                let embedding = self.embed_query(&text).await?;
                let doc = ChromaDocument::new_flat(id.clone(), ns, embedding, metadata, text);
                self.storage.add_to_store(vec![doc]).await?;
                Ok(id)
            }
        }
    }

    pub async fn memory_upsert(
        &self,
        namespace: &str,
        id: String,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<()> {
        // Use Flat mode to preserve the user-provided ID (no onion slicing)
        // This ensures memory_get(id) will find the exact record
        self.index_text_with_mode(Some(namespace), id, text, metadata, SliceMode::Flat)
            .await?;
        Ok(())
    }

    pub async fn memory_get(&self, namespace: &str, id: &str) -> Result<Option<SearchResult>> {
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
        Ok(None)
    }

    pub async fn memory_delete(&self, namespace: &str, id: &str) -> Result<usize> {
        self.storage.delete_document(namespace, id).await
    }

    pub async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        self.storage.purge_namespace(namespace).await
    }

    pub async fn memory_search(
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

        let candidates = self
            .storage
            .search_store_with_layer(
                namespace,
                query_embedding.clone(),
                k * 3,
                options.layer_filter,
            )
            .await?;

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

        // Default: treat as UTF-8 text
        // Path is validated by caller (handlers::validate_path) before reaching this private method
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        tokio::fs::read_to_string(path).await.map_err(|e| e.into())
    }

    /// Extract multiple documents from a JSON file if it contains an array.
    /// For non-array JSON or other file types, returns a single document.
    ///
    /// This enables proper onion slicing for conversation/session files where
    /// each array element (message, conversation) should be indexed separately.
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

        // Only process JSON files specially
        if ext != "json" {
            let text = self.extract_text(path).await?;
            let doc_id = format!("{}:0", path.display());
            let metadata = json!({ "path": path.to_str(), "index": 0 });
            return Ok(vec![(doc_id, text, metadata)]);
        }

        // Try to parse as JSON
        // Path is validated by caller (index_document_with_dedup) via validate_read_path
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        let raw = tokio::fs::read_to_string(path).await?;
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
                    "Conversation array detected: {} -> {} messages",
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
            let match_start = caps.get(0).unwrap().start();

            // Add previous section
            if match_start > last_end {
                let section_content = &content[last_end..match_start];
                if !section_content.trim().is_empty() {
                    sections.push((current_header.clone(), section_content.trim()));
                }
            }

            current_header = Some(caps.get(2).unwrap().as_str().to_string());
            last_end = caps.get(0).unwrap().end();
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
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Filter by onion slice layer (None = all layers)
    pub layer_filter: Option<SliceLayer>,
}

impl SearchOptions {
    /// Search only outer slices (default for onion mode - minimum context, maximum navigation)
    pub fn outer_only() -> Self {
        Self {
            layer_filter: Some(SliceLayer::Outer),
        }
    }

    /// Deep search - include all layers including Core
    pub fn deep() -> Self {
        Self { layer_filter: None }
    }
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
