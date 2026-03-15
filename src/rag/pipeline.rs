//! Async pipeline for concurrent RAG indexing.
//!
//! This module provides an optional pipeline mode where file reading, chunking,
//! embedding, and storage run concurrently using tokio channels.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
//! │ File Reader │ ──► │   Chunker    │ ──► │   Embedder    │ ──► │   Storage   │
//! └─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
//!       tx1                 rx1/tx2              rx2/tx3              rx3
//! ```
//!
//! Each stage runs in its own tokio::spawn, connected via bounded mpsc channels
//! for backpressure. This allows overlapping I/O, CPU, and GPU work.
//!
//! # Usage
//!
//! Pipeline mode is opt-in via the `--pipeline` CLI flag or by calling
//! `index_documents_pipeline()` directly.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::embeddings::EmbeddingClient;
use crate::rag::{
    OnionSlice, OnionSliceConfig, SliceMode, create_onion_slices, create_onion_slices_fast,
};
use crate::storage::{ChromaDocument, StorageManager};

/// Channel buffer size for backpressure.
/// 100 items provides good throughput while limiting memory usage.
const CHANNEL_BUFFER_SIZE: usize = 100;

/// Batch size for storage writes to avoid RAM explosion.
const STORAGE_BATCH_SIZE: usize = 100;

/// File content with metadata for pipeline processing
#[derive(Debug, Clone)]
pub struct FileContent {
    /// Path to the source file
    pub path: PathBuf,
    /// Extracted text content
    pub text: String,
    /// Target namespace for storage
    pub namespace: String,
    /// SHA256 content hash for deduplication
    pub content_hash: String,
}

/// A chunk ready for embedding
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk ID (generated from content)
    pub id: String,
    /// Chunk content text
    pub content: String,
    /// Source file path
    pub source_path: PathBuf,
    /// Target namespace
    pub namespace: String,
    /// Content hash of source file (for dedup tracking)
    pub source_hash: String,
    /// Onion slice layer (if using onion mode)
    pub layer: u8,
    /// Parent slice ID (for onion hierarchy)
    pub parent_id: Option<String>,
    /// Children slice IDs (for onion hierarchy)
    pub children_ids: Vec<String>,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

/// An embedded chunk ready for storage
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    /// Original chunk data
    pub chunk: Chunk,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of files to buffer between reader and chunker
    pub reader_buffer: usize,
    /// Number of chunk batches to buffer between chunker and embedder
    pub chunker_buffer: usize,
    /// Number of embedded batches to buffer between embedder and storage
    pub embedder_buffer: usize,
    /// Slicing mode for chunking
    pub slice_mode: SliceMode,
    /// Enable deduplication
    pub dedup_enabled: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            reader_buffer: CHANNEL_BUFFER_SIZE,
            chunker_buffer: CHANNEL_BUFFER_SIZE,
            embedder_buffer: CHANNEL_BUFFER_SIZE,
            slice_mode: SliceMode::default(),
            dedup_enabled: true,
        }
    }
}

/// Pipeline statistics for progress reporting
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    /// Files read
    pub files_read: usize,
    /// Files skipped (duplicate)
    pub files_skipped: usize,
    /// Chunks created
    pub chunks_created: usize,
    /// Chunks embedded
    pub chunks_embedded: usize,
    /// Chunks stored
    pub chunks_stored: usize,
    /// Errors encountered
    pub errors: usize,
}

/// Result of pipeline execution
#[derive(Debug)]
pub struct PipelineResult {
    /// Pipeline statistics
    pub stats: PipelineStats,
    /// Error messages if any
    pub errors: Vec<String>,
}

// =============================================================================
// STAGE 1: FILE READER
// =============================================================================

/// Stage 1: Read files and extract text content.
///
/// Reads each file, extracts text (supports PDF, text files), and sends
/// to the chunker stage. Handles deduplication check at the storage level.
pub async fn stage_read_files(
    files: Vec<PathBuf>,
    namespace: String,
    storage: Arc<StorageManager>,
    dedup_enabled: bool,
    tx: mpsc::Sender<FileContent>,
) -> (usize, usize) {
    let mut files_read = 0;
    let mut files_skipped = 0;

    for path in files {
        // Extract text from file
        let text = match extract_file_text(&path).await {
            Ok(t) => t,
            Err(e) => {
                warn!("Failed to read file {:?}: {}", path, e);
                continue;
            }
        };

        // Compute content hash
        let content_hash = crate::rag::compute_content_hash(&text);

        // Check for duplicates if enabled
        if dedup_enabled {
            match storage.has_content_hash(&namespace, &content_hash).await {
                Ok(true) => {
                    debug!(
                        "Skipping duplicate: {:?} (hash: {})",
                        path,
                        &content_hash[..16]
                    );
                    files_skipped += 1;
                    continue;
                }
                Ok(false) => {}
                Err(e) => {
                    warn!("Dedup check failed for {:?}: {}", path, e);
                    // Continue anyway - better to index than skip
                }
            }
        }

        let content = FileContent {
            path: path.clone(),
            text,
            namespace: namespace.clone(),
            content_hash,
        };

        // Send to chunker stage
        if tx.send(content).await.is_err() {
            debug!("Reader: channel closed, stopping");
            break;
        }

        files_read += 1;
    }

    info!(
        "Reader stage complete: {} files read, {} skipped",
        files_read, files_skipped
    );
    (files_read, files_skipped)
}

/// Extract text content from a file (PDF or text)
async fn extract_file_text(path: &Path) -> Result<String> {
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

// =============================================================================
// STAGE 2: CHUNKER
// =============================================================================

/// Stage 2: Create chunks/slices from file content.
///
/// Applies the configured slicing mode (onion, onion-fast, flat) to create
/// chunks from each file's content.
pub async fn stage_chunk_content(
    mut rx: mpsc::Receiver<FileContent>,
    tx: mpsc::Sender<Vec<Chunk>>,
    slice_mode: SliceMode,
) -> usize {
    let config = OnionSliceConfig::default();
    let mut total_chunks = 0;

    while let Some(file_content) = rx.recv().await {
        let chunks = create_chunks_from_content(&file_content, slice_mode, &config);
        total_chunks += chunks.len();

        if tx.send(chunks).await.is_err() {
            debug!("Chunker: channel closed, stopping");
            break;
        }
    }

    info!("Chunker stage complete: {} chunks created", total_chunks);
    total_chunks
}

/// Create chunks from file content based on slicing mode
fn create_chunks_from_content(
    content: &FileContent,
    slice_mode: SliceMode,
    config: &OnionSliceConfig,
) -> Vec<Chunk> {
    let metadata = serde_json::json!({
        "path": content.path.to_str(),
        "content_hash": &content.content_hash,
        "slice_mode": match slice_mode {
            SliceMode::Onion => "onion",
            SliceMode::OnionFast => "onion-fast",
            SliceMode::Flat => "flat",
        },
    });

    match slice_mode {
        SliceMode::Onion => {
            let slices = create_onion_slices(&content.text, &metadata, config);
            slices_to_chunks(slices, content)
        }
        SliceMode::OnionFast => {
            let slices = create_onion_slices_fast(&content.text, &metadata, config);
            slices_to_chunks(slices, content)
        }
        SliceMode::Flat => create_flat_chunks(&content.text, content, metadata),
    }
}

/// Convert onion slices to pipeline chunks
fn slices_to_chunks(slices: Vec<OnionSlice>, content: &FileContent) -> Vec<Chunk> {
    slices
        .into_iter()
        .map(|slice| {
            let metadata = serde_json::json!({
                "path": content.path.to_str(),
                "content_hash": &content.content_hash,
                "layer": slice.layer.name(),
            });

            Chunk {
                id: slice.id,
                content: slice.content,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                source_hash: content.content_hash.clone(),
                layer: slice.layer.as_u8(),
                parent_id: slice.parent_id,
                children_ids: slice.children_ids,
                keywords: slice.keywords,
                metadata,
            }
        })
        .collect()
}

/// Create flat chunks from content
fn create_flat_chunks(
    text: &str,
    content: &FileContent,
    base_metadata: serde_json::Value,
) -> Vec<Chunk> {
    let chunks = split_into_chunks(text, 512, 128);
    let total_chunks = chunks.len();

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk_text)| {
            let mut metadata = base_metadata.clone();
            if let serde_json::Value::Object(ref mut map) = metadata {
                map.insert("chunk_index".to_string(), serde_json::json!(idx));
                map.insert("total_chunks".to_string(), serde_json::json!(total_chunks));
            }

            let id = format!(
                "{}_{}_{}",
                content.path.to_str().unwrap_or("unknown"),
                content.content_hash.get(..8).unwrap_or(""),
                idx
            );

            Chunk {
                id,
                content: chunk_text,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                source_hash: content.content_hash.clone(),
                layer: 0, // Flat mode
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
                metadata,
            }
        })
        .collect()
}

/// Simple chunking with overlap
fn split_into_chunks(text: &str, target_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    if len <= target_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let end = (start + target_size).min(len);
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= len {
            break;
        }

        start = end.saturating_sub(overlap);
    }

    chunks
}

// =============================================================================
// STAGE 3: EMBEDDER
// =============================================================================

/// Stage 3: Embed chunks using the embedding client.
///
/// Processes batches of chunks, generates embeddings, and forwards to storage.
/// Handles batch size limits and error recovery.
pub async fn stage_embed_chunks(
    mut rx: mpsc::Receiver<Vec<Chunk>>,
    tx: mpsc::Sender<Vec<EmbeddedChunk>>,
    client: Arc<Mutex<EmbeddingClient>>,
) -> (usize, usize) {
    let mut total_embedded = 0;
    let mut errors = 0;

    while let Some(chunks) = rx.recv().await {
        if chunks.is_empty() {
            continue;
        }

        // Extract texts for embedding
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        // Get embeddings
        let embeddings = match client.lock().await.embed_batch(&texts).await {
            Ok(embs) => embs,
            Err(e) => {
                error!("Embedding batch failed: {}", e);
                errors += chunks.len();
                continue;
            }
        };

        // Combine chunks with their embeddings
        let embedded: Vec<EmbeddedChunk> = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| EmbeddedChunk { chunk, embedding })
            .collect();

        total_embedded += embedded.len();

        if tx.send(embedded).await.is_err() {
            debug!("Embedder: channel closed, stopping");
            break;
        }
    }

    info!(
        "Embedder stage complete: {} chunks embedded, {} errors",
        total_embedded, errors
    );
    (total_embedded, errors)
}

// =============================================================================
// STAGE 4: STORAGE
// =============================================================================

/// Stage 4: Store embedded chunks to the database.
///
/// Batches incoming embedded chunks and writes to storage efficiently.
pub async fn stage_store_chunks(
    mut rx: mpsc::Receiver<Vec<EmbeddedChunk>>,
    storage: Arc<StorageManager>,
) -> usize {
    let mut total_stored = 0;
    let mut buffer: Vec<EmbeddedChunk> = Vec::new();

    while let Some(embedded_chunks) = rx.recv().await {
        buffer.extend(embedded_chunks);

        // Flush when buffer exceeds batch size
        while buffer.len() >= STORAGE_BATCH_SIZE {
            let batch: Vec<EmbeddedChunk> = buffer.drain(..STORAGE_BATCH_SIZE).collect();
            match store_batch(&storage, batch).await {
                Ok(count) => total_stored += count,
                Err(e) => error!("Storage batch failed: {}", e),
            }
        }
    }

    // Flush remaining items
    if !buffer.is_empty() {
        match store_batch(&storage, buffer).await {
            Ok(count) => total_stored += count,
            Err(e) => error!("Final storage batch failed: {}", e),
        }
    }

    info!("Storage stage complete: {} chunks stored", total_stored);
    total_stored
}

/// Store a batch of embedded chunks
async fn store_batch(storage: &StorageManager, batch: Vec<EmbeddedChunk>) -> Result<usize> {
    let count = batch.len();

    let documents: Vec<ChromaDocument> = batch
        .into_iter()
        .map(|ec| {
            if ec.chunk.layer > 0 {
                // Onion slice mode
                ChromaDocument {
                    id: ec.chunk.id,
                    namespace: ec.chunk.namespace,
                    embedding: ec.embedding,
                    metadata: ec.chunk.metadata,
                    document: ec.chunk.content,
                    layer: ec.chunk.layer,
                    parent_id: ec.chunk.parent_id,
                    children_ids: ec.chunk.children_ids,
                    keywords: ec.chunk.keywords,
                    content_hash: Some(ec.chunk.source_hash),
                }
            } else {
                // Flat mode
                ChromaDocument::new_flat_with_hash(
                    ec.chunk.id,
                    ec.chunk.namespace,
                    ec.embedding,
                    ec.chunk.metadata,
                    ec.chunk.content,
                    ec.chunk.source_hash,
                )
            }
        })
        .collect();

    storage.add_to_store(documents).await?;
    debug!("Stored batch of {} chunks", count);
    Ok(count)
}

// =============================================================================
// PIPELINE COORDINATOR
// =============================================================================

/// Run the async pipeline for document indexing.
///
/// Spawns all stages concurrently and waits for completion.
pub async fn run_pipeline(
    files: Vec<PathBuf>,
    namespace: String,
    storage: Arc<StorageManager>,
    client: Arc<Mutex<EmbeddingClient>>,
    config: PipelineConfig,
) -> Result<PipelineResult> {
    let total_files = files.len();
    info!(
        "Starting pipeline: {} files, mode: {:?}",
        total_files, config.slice_mode
    );

    // Create channels
    let (tx1, rx1) = mpsc::channel::<FileContent>(config.reader_buffer);
    let (tx2, rx2) = mpsc::channel::<Vec<Chunk>>(config.chunker_buffer);
    let (tx3, rx3) = mpsc::channel::<Vec<EmbeddedChunk>>(config.embedder_buffer);

    // Clone references for each stage
    let storage_for_reader = storage.clone();
    let storage_for_storage = storage;
    let ns_for_reader = namespace.clone();
    let slice_mode = config.slice_mode;
    let dedup_enabled = config.dedup_enabled;

    // Spawn all stages
    let reader_handle = tokio::spawn(async move {
        stage_read_files(files, ns_for_reader, storage_for_reader, dedup_enabled, tx1).await
    });

    let chunker_handle =
        tokio::spawn(async move { stage_chunk_content(rx1, tx2, slice_mode).await });

    let embedder_handle = tokio::spawn(async move { stage_embed_chunks(rx2, tx3, client).await });

    let storage_handle =
        tokio::spawn(async move { stage_store_chunks(rx3, storage_for_storage).await });

    // Wait for all stages to complete
    let (reader_result, chunker_result, embedder_result, storage_result) = tokio::try_join!(
        reader_handle,
        chunker_handle,
        embedder_handle,
        storage_handle
    )?;

    let (files_read, files_skipped) = reader_result;
    let chunks_created = chunker_result;
    let (chunks_embedded, embed_errors) = embedder_result;
    let chunks_stored = storage_result;

    let stats = PipelineStats {
        files_read,
        files_skipped,
        chunks_created,
        chunks_embedded,
        chunks_stored,
        errors: embed_errors,
    };

    info!(
        "Pipeline complete: {} files -> {} chunks -> {} stored",
        files_read, chunks_created, chunks_stored
    );

    Ok(PipelineResult {
        stats,
        errors: vec![], // TODO: collect error messages
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_into_chunks_short_text() {
        let text = "Hello world";
        let chunks = split_into_chunks(text, 100, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_split_into_chunks_with_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = split_into_chunks(text, 10, 3);
        assert!(chunks.len() > 1);
        // First chunk should be 10 chars
        assert_eq!(chunks[0].len(), 10);
        // Chunks should overlap
        assert!(chunks[0].ends_with(&chunks[1][..3]));
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.reader_buffer, CHANNEL_BUFFER_SIZE);
        assert_eq!(config.slice_mode, SliceMode::default());
        assert!(config.dedup_enabled);
    }
}
