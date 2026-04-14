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
//! Pipeline mode is opt-in via the `--pipeline` CLI flag or by calling
//! `run_pipeline()` directly.

use anyhow::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc};
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

/// File content with metadata for pipeline processing.
#[derive(Debug, Clone)]
pub struct FileContent {
    /// Path to the source file.
    pub path: PathBuf,
    /// Extracted text content.
    pub text: String,
    /// Target namespace for storage.
    pub namespace: String,
    /// SHA256 content hash for deduplication.
    pub content_hash: String,
}

/// A chunk ready for embedding.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk ID (generated from content).
    pub id: String,
    /// Chunk content text.
    pub content: String,
    /// Source file path.
    pub source_path: PathBuf,
    /// Target namespace.
    pub namespace: String,
    /// Content hash of source file (for dedup tracking).
    pub source_hash: String,
    /// Onion slice layer (if using onion mode).
    pub layer: u8,
    /// Parent slice ID (for onion hierarchy).
    pub parent_id: Option<String>,
    /// Children slice IDs (for onion hierarchy).
    pub children_ids: Vec<String>,
    /// Extracted keywords.
    pub keywords: Vec<String>,
    /// Additional metadata.
    pub metadata: serde_json::Value,
}

/// An embedded chunk ready for storage.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    /// Original chunk data.
    pub chunk: Chunk,
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ChunkBatch {
    path: PathBuf,
    content_hash: String,
    chunks: Vec<Chunk>,
}

#[derive(Debug, Clone)]
struct EmbeddedFile {
    path: PathBuf,
    content_hash: String,
    chunks: Vec<EmbeddedChunk>,
}

#[derive(Debug, Clone, Copy)]
struct EmbedRuntimeSettings {
    max_batch_chars: usize,
    max_batch_items: usize,
    concurrency: usize,
}

impl EmbedRuntimeSettings {
    fn new(max_batch_chars: usize, max_batch_items: usize, concurrency: usize) -> Self {
        Self {
            max_batch_chars: max_batch_chars.max(1),
            max_batch_items: max_batch_items.max(1),
            concurrency: concurrency.max(1),
        }
    }
}

/// Adaptive governor envelope for pipeline embedding throughput.
#[derive(Debug, Clone)]
pub struct PipelineGovernorConfig {
    pub min_batch_chars: usize,
    pub max_batch_chars: usize,
    pub min_batch_items: usize,
    pub max_batch_items: usize,
    pub min_concurrency: usize,
    pub max_concurrency: usize,
    pub target_latency: Duration,
    pub pressure_latency: Duration,
    pub growth_cooldown: Duration,
    pub pressure_cooldown: Duration,
    pub backlog_low_watermark: usize,
    pub storage_backlog_high_watermark: usize,
}

impl PipelineGovernorConfig {
    pub fn adaptive(
        max_batch_chars: usize,
        max_batch_items: usize,
        max_concurrency: usize,
    ) -> Self {
        let max_batch_chars = max_batch_chars.max(1);
        let max_batch_items = max_batch_items.max(1);
        let max_concurrency = max_concurrency.max(1);

        Self {
            min_batch_chars: (max_batch_chars / 4).max(4_096).min(max_batch_chars),
            max_batch_chars,
            min_batch_items: (max_batch_items / 4).max(1).min(max_batch_items),
            max_batch_items,
            min_concurrency: 1,
            max_concurrency,
            target_latency: Duration::from_millis(900),
            pressure_latency: Duration::from_millis(2_200),
            growth_cooldown: Duration::from_secs(3),
            pressure_cooldown: Duration::from_millis(750),
            backlog_low_watermark: 2,
            storage_backlog_high_watermark: 3,
        }
    }

    fn initial_settings(&self) -> EmbedRuntimeSettings {
        EmbedRuntimeSettings::new(
            self.min_batch_chars,
            self.min_batch_items,
            self.min_concurrency,
        )
    }
}

#[derive(Debug, Clone)]
struct PipelineGovernorAdjustment {
    settings: EmbedRuntimeSettings,
    mode: String,
    reason: String,
}

#[derive(Debug, Clone)]
struct PipelineGovernor {
    config: PipelineGovernorConfig,
    current: EmbedRuntimeSettings,
    last_growth_at: Option<Instant>,
    last_pressure_at: Option<Instant>,
}

impl PipelineGovernor {
    fn new(config: PipelineGovernorConfig) -> Self {
        let current = config.initial_settings();
        Self {
            config,
            current,
            last_growth_at: None,
            last_pressure_at: None,
        }
    }

    fn current_settings(&self) -> EmbedRuntimeSettings {
        self.current
    }

    fn initial_adjustment(&self) -> PipelineGovernorAdjustment {
        PipelineGovernorAdjustment {
            settings: self.current,
            mode: "adaptive".to_string(),
            reason: "warming up from conservative limits".to_string(),
        }
    }

    fn on_success(
        &mut self,
        elapsed: Duration,
        snapshot: &PipelineSnapshot,
    ) -> Option<PipelineGovernorAdjustment> {
        let backlog = snapshot.chunker_queue_depth + snapshot.reader_queue_depth;
        let storage_pressure =
            snapshot.storage_queue_depth >= self.config.storage_backlog_high_watermark;
        let slow_batch = elapsed >= self.config.pressure_latency;

        if (storage_pressure || slow_batch) && self.pressure_ready() {
            let mut changed = false;
            let mut reasons = Vec::new();

            if self.current.max_batch_items > self.config.min_batch_items {
                let next_items =
                    ((self.current.max_batch_items * 2) / 3).max(self.config.min_batch_items);
                if next_items != self.current.max_batch_items {
                    self.current.max_batch_items = next_items;
                    changed = true;
                    reasons.push(format!("items {}", next_items));
                }
            }

            if self.current.max_batch_chars > self.config.min_batch_chars {
                let next_chars =
                    ((self.current.max_batch_chars * 2) / 3).max(self.config.min_batch_chars);
                if next_chars != self.current.max_batch_chars {
                    self.current.max_batch_chars = next_chars;
                    changed = true;
                    reasons.push(format!("chars {}", next_chars));
                }
            }

            if storage_pressure && self.current.concurrency > self.config.min_concurrency {
                self.current.concurrency -= 1;
                changed = true;
                reasons.push(format!("concurrency {}", self.current.concurrency));
            }

            if changed {
                self.last_pressure_at = Some(Instant::now());
                let reason = if storage_pressure {
                    format!(
                        "pressure: storage backlog {} -> {}",
                        snapshot.storage_queue_depth,
                        reasons.join(", ")
                    )
                } else {
                    format!(
                        "pressure: embed {:.0}ms -> {}",
                        elapsed.as_secs_f64() * 1_000.0,
                        reasons.join(", ")
                    )
                };
                return Some(PipelineGovernorAdjustment {
                    settings: self.current,
                    mode: "adaptive".to_string(),
                    reason,
                });
            }
        }

        if elapsed > self.config.target_latency
            || backlog < self.config.backlog_low_watermark
            || snapshot.storage_queue_depth > 1
            || !self.growth_ready()
        {
            return None;
        }

        if self.current.max_batch_items < self.config.max_batch_items {
            let step = (self.current.max_batch_items / 4).max(1);
            let next_items = (self.current.max_batch_items + step).min(self.config.max_batch_items);
            if next_items != self.current.max_batch_items {
                self.current.max_batch_items = next_items;
                self.last_growth_at = Some(Instant::now());
                return Some(PipelineGovernorAdjustment {
                    settings: self.current,
                    mode: "adaptive".to_string(),
                    reason: format!(
                        "backlog {} with fast embed {:.0}ms -> items {}",
                        backlog,
                        elapsed.as_secs_f64() * 1_000.0,
                        next_items
                    ),
                });
            }
        }

        if self.current.max_batch_chars < self.config.max_batch_chars {
            let step = (self.current.max_batch_chars / 4).max(4_096);
            let next_chars = (self.current.max_batch_chars + step).min(self.config.max_batch_chars);
            if next_chars != self.current.max_batch_chars {
                self.current.max_batch_chars = next_chars;
                self.last_growth_at = Some(Instant::now());
                return Some(PipelineGovernorAdjustment {
                    settings: self.current,
                    mode: "adaptive".to_string(),
                    reason: format!(
                        "backlog {} with fast embed {:.0}ms -> chars {}",
                        backlog,
                        elapsed.as_secs_f64() * 1_000.0,
                        next_chars
                    ),
                });
            }
        }

        let concurrency_backlog = self
            .current
            .concurrency
            .saturating_mul(2)
            .max(self.config.backlog_low_watermark);
        if backlog >= concurrency_backlog && self.current.concurrency < self.config.max_concurrency
        {
            self.current.concurrency += 1;
            self.last_growth_at = Some(Instant::now());
            return Some(PipelineGovernorAdjustment {
                settings: self.current,
                mode: "adaptive".to_string(),
                reason: format!(
                    "backlog {} sustained with fast embed {:.0}ms -> concurrency {}",
                    backlog,
                    elapsed.as_secs_f64() * 1_000.0,
                    self.current.concurrency
                ),
            });
        }

        None
    }

    fn on_error(
        &mut self,
        snapshot: &PipelineSnapshot,
        message: &str,
    ) -> Option<PipelineGovernorAdjustment> {
        if !self.pressure_ready() {
            return None;
        }

        let mut changed = false;

        if self.current.concurrency > self.config.min_concurrency {
            self.current.concurrency = self.config.min_concurrency;
            changed = true;
        }
        if self.current.max_batch_items > self.config.min_batch_items {
            self.current.max_batch_items = self.config.min_batch_items;
            changed = true;
        }
        if self.current.max_batch_chars > self.config.min_batch_chars {
            self.current.max_batch_chars = self.config.min_batch_chars;
            changed = true;
        }

        if !changed {
            return None;
        }

        self.last_pressure_at = Some(Instant::now());
        Some(PipelineGovernorAdjustment {
            settings: self.current,
            mode: "adaptive".to_string(),
            reason: format!(
                "error with backlog {} and storage {}: {}",
                snapshot.chunker_queue_depth + snapshot.reader_queue_depth,
                snapshot.storage_queue_depth,
                message
            ),
        })
    }

    fn growth_ready(&self) -> bool {
        self.last_growth_at
            .map(|instant| instant.elapsed() >= self.config.growth_cooldown)
            .unwrap_or(true)
    }

    fn pressure_ready(&self) -> bool {
        self.last_pressure_at
            .map(|instant| instant.elapsed() >= self.config.pressure_cooldown)
            .unwrap_or(true)
    }
}

#[derive(Debug)]
struct EmbedWorkerResult {
    path: PathBuf,
    content_hash: String,
    count: usize,
    chars: usize,
    elapsed: Duration,
    chunks: Option<Vec<EmbeddedChunk>>,
    error: Option<String>,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of files to buffer between reader and chunker.
    pub reader_buffer: usize,
    /// Number of chunk batches to buffer between chunker and embedder.
    pub chunker_buffer: usize,
    /// Number of embedded batches to buffer between embedder and storage.
    pub embedder_buffer: usize,
    /// Slicing mode for chunking.
    pub slice_mode: SliceMode,
    /// Enable storage-backed deduplication.
    pub dedup_enabled: bool,
    /// Maximum number of embedding requests allowed in flight.
    pub embed_concurrency: usize,
    /// Optional adaptive governor for runtime batch/concurrency tuning.
    pub governor: Option<PipelineGovernorConfig>,
    /// Optional event stream for progress/reporting consumers.
    pub event_sender: Option<mpsc::UnboundedSender<PipelineEvent>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            reader_buffer: CHANNEL_BUFFER_SIZE,
            chunker_buffer: CHANNEL_BUFFER_SIZE,
            embedder_buffer: CHANNEL_BUFFER_SIZE,
            slice_mode: SliceMode::default(),
            dedup_enabled: true,
            embed_concurrency: 1,
            governor: None,
            event_sender: None,
        }
    }
}

/// Pipeline statistics for progress reporting.
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    /// Total files scheduled for this pipeline run.
    pub total_files: usize,
    /// Files successfully read and handed to the chunker.
    pub files_read: usize,
    /// Files skipped before chunking (for example exact duplicates).
    pub files_skipped: usize,
    /// Files durably committed to storage.
    pub files_committed: usize,
    /// Files that failed in any stage.
    pub files_failed: usize,
    /// Chunks created.
    pub chunks_created: usize,
    /// Chunks embedded.
    pub chunks_embedded: usize,
    /// Chunks durably stored.
    pub chunks_stored: usize,
    /// Stage-local errors encountered.
    pub errors: usize,
}

/// Runtime snapshot for live progress consumers.
#[derive(Debug, Clone, Default)]
pub struct PipelineSnapshot {
    pub total_files: usize,
    pub files_read: usize,
    pub files_skipped: usize,
    pub files_committed: usize,
    pub files_failed: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub chunks_stored: usize,
    pub errors: usize,
    pub reader_queue_depth: usize,
    pub chunker_queue_depth: usize,
    pub storage_queue_depth: usize,
    pub current_embed_batch_items: usize,
    pub current_embed_batch_chars: usize,
    pub embed_batch_items_limit: usize,
    pub embed_batch_chars_limit: usize,
    pub embed_active_requests: usize,
    pub embed_concurrency_limit: usize,
    pub avg_embed_batch_ms: Option<f64>,
    pub files_per_sec: f64,
    pub chunks_per_sec: f64,
    pub eta: Option<Duration>,
    pub elapsed: Duration,
    pub bottleneck: String,
    pub governor_mode: String,
    pub governor_reason: String,
}

impl PipelineSnapshot {
    pub fn to_stats(&self) -> PipelineStats {
        PipelineStats {
            total_files: self.total_files,
            files_read: self.files_read,
            files_skipped: self.files_skipped,
            files_committed: self.files_committed,
            files_failed: self.files_failed,
            chunks_created: self.chunks_created,
            chunks_embedded: self.chunks_embedded,
            chunks_stored: self.chunks_stored,
            errors: self.errors,
        }
    }
}

/// Progress and lifecycle events emitted by the pipeline.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    FileRead {
        path: PathBuf,
        content_hash: String,
        bytes: usize,
    },
    FileSkipped {
        path: PathBuf,
        content_hash: String,
        reason: String,
    },
    ChunksCreated {
        path: PathBuf,
        content_hash: String,
        count: usize,
    },
    EmbedStarted {
        path: PathBuf,
        content_hash: String,
        count: usize,
        chars: usize,
        batch_items_limit: usize,
        batch_chars_limit: usize,
        concurrency_limit: usize,
    },
    ChunksEmbedded {
        path: PathBuf,
        content_hash: String,
        count: usize,
        chars: usize,
        elapsed: Duration,
    },
    GovernorAdjusted {
        batch_items_limit: usize,
        batch_chars_limit: usize,
        concurrency_limit: usize,
        mode: String,
        reason: String,
    },
    FileCommitted {
        path: PathBuf,
        content_hash: String,
        chunk_count: usize,
    },
    Error {
        path: Option<PathBuf>,
        stage: &'static str,
        message: String,
    },
    Snapshot(PipelineSnapshot),
}

/// Result of pipeline execution.
#[derive(Debug)]
pub struct PipelineResult {
    /// Pipeline statistics.
    pub stats: PipelineStats,
    /// Error messages if any.
    pub errors: Vec<String>,
}

#[derive(Debug)]
struct PipelineProgressState {
    snapshot: PipelineSnapshot,
    started_at: Instant,
    failed_paths: HashSet<String>,
    error_messages: Vec<String>,
}

impl PipelineProgressState {
    fn new(
        total_files: usize,
        initial_runtime: EmbedRuntimeSettings,
        governor_mode: String,
        governor_reason: String,
    ) -> Self {
        let snapshot = PipelineSnapshot {
            total_files,
            embed_batch_items_limit: initial_runtime.max_batch_items,
            embed_batch_chars_limit: initial_runtime.max_batch_chars,
            embed_concurrency_limit: initial_runtime.concurrency,
            bottleneck: "reader".to_string(),
            governor_mode,
            governor_reason,
            ..Default::default()
        };

        Self {
            snapshot,
            started_at: Instant::now(),
            failed_paths: HashSet::new(),
            error_messages: Vec::new(),
        }
    }

    fn record_failure(&mut self, path: &Path) {
        let key = path.to_string_lossy().to_string();
        if self.failed_paths.insert(key) {
            self.snapshot.files_failed += 1;
        }
    }

    fn apply(&mut self, event: &PipelineEvent) -> PipelineSnapshot {
        match event {
            PipelineEvent::FileRead { .. } => {
                self.snapshot.files_read += 1;
                self.snapshot.reader_queue_depth += 1;
            }
            PipelineEvent::FileSkipped { .. } => {
                self.snapshot.files_skipped += 1;
            }
            PipelineEvent::ChunksCreated { count, .. } => {
                self.snapshot.chunks_created += count;
                self.snapshot.reader_queue_depth =
                    self.snapshot.reader_queue_depth.saturating_sub(1);
                self.snapshot.chunker_queue_depth += 1;
            }
            PipelineEvent::EmbedStarted {
                batch_items_limit,
                batch_chars_limit,
                concurrency_limit,
                ..
            } => {
                self.snapshot.embed_batch_items_limit = *batch_items_limit;
                self.snapshot.embed_batch_chars_limit = *batch_chars_limit;
                self.snapshot.embed_concurrency_limit = *concurrency_limit;
                self.snapshot.embed_active_requests += 1;
            }
            PipelineEvent::ChunksEmbedded {
                count,
                chars,
                elapsed,
                ..
            } => {
                self.snapshot.chunks_embedded += count;
                self.snapshot.chunker_queue_depth =
                    self.snapshot.chunker_queue_depth.saturating_sub(1);
                self.snapshot.storage_queue_depth += 1;
                self.snapshot.embed_active_requests =
                    self.snapshot.embed_active_requests.saturating_sub(1);
                self.snapshot.current_embed_batch_items = *count;
                self.snapshot.current_embed_batch_chars = *chars;
                let latency_ms = elapsed.as_secs_f64() * 1_000.0;
                self.snapshot.avg_embed_batch_ms = Some(
                    self.snapshot
                        .avg_embed_batch_ms
                        .map(|existing| (existing * 0.7) + (latency_ms * 0.3))
                        .unwrap_or(latency_ms),
                );
            }
            PipelineEvent::GovernorAdjusted {
                batch_items_limit,
                batch_chars_limit,
                concurrency_limit,
                mode,
                reason,
            } => {
                self.snapshot.embed_batch_items_limit = *batch_items_limit;
                self.snapshot.embed_batch_chars_limit = *batch_chars_limit;
                self.snapshot.embed_concurrency_limit = *concurrency_limit;
                self.snapshot.governor_mode = mode.clone();
                self.snapshot.governor_reason = reason.clone();
            }
            PipelineEvent::FileCommitted { chunk_count, .. } => {
                self.snapshot.files_committed += 1;
                self.snapshot.chunks_stored += chunk_count;
                self.snapshot.storage_queue_depth =
                    self.snapshot.storage_queue_depth.saturating_sub(1);
            }
            PipelineEvent::Error {
                path,
                stage,
                message,
            } => {
                self.snapshot.errors += 1;
                match *stage {
                    "embedder" => {
                        self.snapshot.chunker_queue_depth =
                            self.snapshot.chunker_queue_depth.saturating_sub(1);
                        self.snapshot.embed_active_requests =
                            self.snapshot.embed_active_requests.saturating_sub(1);
                    }
                    "storage" => {
                        self.snapshot.storage_queue_depth =
                            self.snapshot.storage_queue_depth.saturating_sub(1);
                    }
                    _ => {}
                }
                if let Some(path) = path {
                    self.record_failure(path);
                }
                self.error_messages.push(match path {
                    Some(path) => format!("{} [{}]: {}", stage, path.display(), message),
                    None => format!("{}: {}", stage, message),
                });
            }
            PipelineEvent::Snapshot(snapshot) => {
                self.snapshot = snapshot.clone();
            }
        }

        self.refresh_snapshot()
    }

    fn refresh_snapshot(&mut self) -> PipelineSnapshot {
        let elapsed = self.started_at.elapsed();
        let terminal_files = self.snapshot.files_committed
            + self.snapshot.files_skipped
            + self.snapshot.files_failed;
        let elapsed_secs = elapsed.as_secs_f64();

        self.snapshot.elapsed = elapsed;
        self.snapshot.files_per_sec = if elapsed_secs > 0.0 {
            terminal_files as f64 / elapsed_secs
        } else {
            0.0
        };
        self.snapshot.chunks_per_sec = if elapsed_secs > 0.0 {
            self.snapshot.chunks_stored as f64 / elapsed_secs
        } else {
            0.0
        };

        let remaining_files = self.snapshot.total_files.saturating_sub(terminal_files);
        self.snapshot.eta = if self.snapshot.files_per_sec > 0.0 && remaining_files > 0 {
            Some(Duration::from_secs_f64(
                remaining_files as f64 / self.snapshot.files_per_sec,
            ))
        } else {
            None
        };

        self.snapshot.bottleneck = determine_bottleneck(&self.snapshot);
        self.snapshot.clone()
    }
}

fn determine_bottleneck(snapshot: &PipelineSnapshot) -> String {
    let terminal_files = snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
    if terminal_files >= snapshot.total_files {
        return "complete".to_string();
    }

    let mut stage = "idle";
    let mut depth = 0usize;

    if snapshot.reader_queue_depth > depth {
        stage = "chunker";
        depth = snapshot.reader_queue_depth;
    }
    if snapshot.chunker_queue_depth > depth {
        stage = "embedder";
        depth = snapshot.chunker_queue_depth;
    }
    if snapshot.storage_queue_depth > depth {
        stage = "storage";
        depth = snapshot.storage_queue_depth;
    }
    if snapshot.embed_active_requests > 0
        && snapshot.embed_active_requests >= snapshot.embed_concurrency_limit.max(1)
        && depth <= snapshot.chunker_queue_depth
    {
        stage = "embedder";
        depth = snapshot.embed_active_requests;
    }
    if depth == 0 && snapshot.files_read < snapshot.total_files {
        stage = "reader";
    }

    stage.to_string()
}

#[derive(Clone)]
struct PipelineObserver {
    sender: Option<mpsc::UnboundedSender<PipelineEvent>>,
    state: Arc<Mutex<PipelineProgressState>>,
}

impl PipelineObserver {
    fn new(
        total_files: usize,
        sender: Option<mpsc::UnboundedSender<PipelineEvent>>,
        initial_runtime: EmbedRuntimeSettings,
        governor_mode: String,
        governor_reason: String,
    ) -> Self {
        Self {
            sender,
            state: Arc::new(Mutex::new(PipelineProgressState::new(
                total_files,
                initial_runtime,
                governor_mode,
                governor_reason,
            ))),
        }
    }

    async fn emit(&self, event: PipelineEvent) {
        let snapshot = {
            let mut state = self.state.lock().await;
            state.apply(&event)
        };

        if let Some(sender) = &self.sender {
            let _ = sender.send(event);
            let _ = sender.send(PipelineEvent::Snapshot(snapshot));
        }
    }

    async fn emit_initial_snapshot(&self) {
        let snapshot = {
            let mut state = self.state.lock().await;
            state.refresh_snapshot()
        };

        if let Some(sender) = &self.sender {
            let _ = sender.send(PipelineEvent::Snapshot(snapshot));
        }
    }

    async fn result(&self) -> PipelineResult {
        let state = self.state.lock().await;
        PipelineResult {
            stats: state.snapshot.to_stats(),
            errors: state.error_messages.clone(),
        }
    }

    async fn snapshot(&self) -> PipelineSnapshot {
        let state = self.state.lock().await;
        state.snapshot.clone()
    }
}

// =============================================================================
// STAGE 1: FILE READER
// =============================================================================

/// Stage 1: Read files and extract text content.
///
/// Reads each file, extracts text (supports PDF, text files), and sends
/// to the chunker stage. Handles deduplication check at the storage level.
async fn stage_read_files(
    files: Vec<PathBuf>,
    namespace: String,
    storage: Arc<StorageManager>,
    dedup_enabled: bool,
    tx: mpsc::Sender<FileContent>,
    observer: PipelineObserver,
) {
    for path in files {
        let text = match extract_file_text(&path).await {
            Ok(text) => text,
            Err(err) => {
                warn!("Failed to read file {:?}: {}", path, err);
                observer
                    .emit(PipelineEvent::Error {
                        path: Some(path.clone()),
                        stage: "reader",
                        message: err.to_string(),
                    })
                    .await;
                continue;
            }
        };

        let content_hash = crate::rag::compute_content_hash(&text);

        if dedup_enabled {
            match storage.has_content_hash(&namespace, &content_hash).await {
                Ok(true) => {
                    debug!(
                        "Skipping duplicate: {:?} (hash: {})",
                        path,
                        &content_hash[..16]
                    );
                    observer
                        .emit(PipelineEvent::FileSkipped {
                            path: path.clone(),
                            content_hash,
                            reason: "exact duplicate".to_string(),
                        })
                        .await;
                    continue;
                }
                Ok(false) => {}
                Err(err) => {
                    warn!("Dedup check failed for {:?}: {}", path, err);
                }
            }
        }

        let bytes = text.len();
        let content = FileContent {
            path: path.clone(),
            text,
            namespace: namespace.clone(),
            content_hash: content_hash.clone(),
        };

        if tx.send(content).await.is_err() {
            debug!("Reader: channel closed, stopping");
            break;
        }

        observer
            .emit(PipelineEvent::FileRead {
                path,
                content_hash,
                bytes,
            })
            .await;
    }

    info!("Reader stage complete");
}

/// Extract text content from a file (PDF or text).
async fn extract_file_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "pdf" {
        let path = path.to_path_buf();
        let pdf_text =
            tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path)).await??;
        return Ok(pdf_text);
    }

    let (_path, content) = crate::path_utils::safe_read_to_string_async(path).await?;
    Ok(content)
}

// =============================================================================
// STAGE 2: CHUNKER
// =============================================================================

/// Stage 2: Create chunks/slices from file content.
async fn stage_chunk_content(
    mut rx: mpsc::Receiver<FileContent>,
    tx: mpsc::Sender<ChunkBatch>,
    slice_mode: SliceMode,
    observer: PipelineObserver,
) {
    let config = OnionSliceConfig::default();

    while let Some(file_content) = rx.recv().await {
        let path = file_content.path.clone();
        let content_hash = file_content.content_hash.clone();
        let chunks = create_chunks_from_content(&file_content, slice_mode, &config);
        let count = chunks.len();

        if tx
            .send(ChunkBatch {
                path: path.clone(),
                content_hash: content_hash.clone(),
                chunks,
            })
            .await
            .is_err()
        {
            debug!("Chunker: channel closed, stopping");
            break;
        }

        observer
            .emit(PipelineEvent::ChunksCreated {
                path,
                content_hash,
                count,
            })
            .await;
    }

    info!("Chunker stage complete");
}

/// Create chunks from file content based on slicing mode.
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

/// Convert onion slices to pipeline chunks.
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

/// Create flat chunks from content.
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
                layer: 0,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
                metadata,
            }
        })
        .collect()
}

/// Simple chunking with overlap.
fn split_into_chunks(text: &str, target_size: usize, overlap: usize) -> Vec<String> {
    let mut char_offsets: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let len = char_offsets.len();

    if len <= target_size {
        return vec![text.to_string()];
    }

    char_offsets.push(text.len());

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let end = (start + target_size).min(len);
        let start_byte = char_offsets[start];
        let end_byte = char_offsets[end];
        chunks.push(text[start_byte..end_byte].to_string());

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
async fn stage_embed_chunks(
    mut rx: mpsc::Receiver<ChunkBatch>,
    tx: mpsc::Sender<EmbeddedFile>,
    base_client: EmbeddingClient,
    embed_concurrency: usize,
    governor_config: Option<PipelineGovernorConfig>,
    observer: PipelineObserver,
) {
    let fixed_settings = {
        let (max_batch_chars, max_batch_items) = base_client.batch_limits();
        EmbedRuntimeSettings::new(max_batch_chars, max_batch_items, embed_concurrency)
    };
    let mut governor = governor_config.map(PipelineGovernor::new);
    let initial_adjustment = governor
        .as_ref()
        .map(PipelineGovernor::initial_adjustment)
        .unwrap_or_else(|| PipelineGovernorAdjustment {
            settings: fixed_settings,
            mode: "fixed".to_string(),
            reason: "operator-configured limits".to_string(),
        });
    observer
        .emit(PipelineEvent::GovernorAdjusted {
            batch_items_limit: initial_adjustment.settings.max_batch_items,
            batch_chars_limit: initial_adjustment.settings.max_batch_chars,
            concurrency_limit: initial_adjustment.settings.concurrency,
            mode: initial_adjustment.mode,
            reason: initial_adjustment.reason,
        })
        .await;

    let result_capacity = embed_concurrency.max(1).saturating_mul(2);
    let (result_tx, mut result_rx) = mpsc::channel::<EmbedWorkerResult>(result_capacity.max(2));
    let mut input_closed = false;
    let mut in_flight = 0usize;

    loop {
        if input_closed && in_flight == 0 {
            break;
        }

        let settings = governor
            .as_ref()
            .map(PipelineGovernor::current_settings)
            .unwrap_or(fixed_settings);

        if !input_closed && in_flight < settings.concurrency {
            tokio::select! {
                maybe_batch = rx.recv() => {
                    match maybe_batch {
                        Some(chunk_batch) => {
                            if chunk_batch.chunks.is_empty() {
                                continue;
                            }

                            let batch_chars: usize = chunk_batch
                                .chunks
                                .iter()
                                .map(|chunk| chunk.content.chars().count())
                                .sum();
                            observer
                                .emit(PipelineEvent::EmbedStarted {
                                    path: chunk_batch.path.clone(),
                                    content_hash: chunk_batch.content_hash.clone(),
                                    count: chunk_batch.chunks.len(),
                                    chars: batch_chars,
                                    batch_items_limit: settings.max_batch_items,
                                    batch_chars_limit: settings.max_batch_chars,
                                    concurrency_limit: settings.concurrency,
                                })
                                .await;

                            in_flight += 1;
                            let worker_tx = result_tx.clone();
                            let worker_client = base_client
                                .clone_with_batch_limits(settings.max_batch_chars, settings.max_batch_items);
                            tokio::spawn(async move {
                                let result = embed_chunk_batch(worker_client, chunk_batch).await;
                                let _ = worker_tx.send(result).await;
                            });
                        }
                        None => input_closed = true,
                    }
                }
                Some(result) = result_rx.recv(), if in_flight > 0 => {
                    in_flight = in_flight.saturating_sub(1);
                    if !handle_embed_result(result, &tx, &observer, governor.as_mut()).await {
                        break;
                    }
                }
            }
        } else if let Some(result) = result_rx.recv().await {
            in_flight = in_flight.saturating_sub(1);
            if !handle_embed_result(result, &tx, &observer, governor.as_mut()).await {
                break;
            }
        } else {
            break;
        }
    }

    info!("Embedder stage complete");
}

async fn embed_chunk_batch(
    mut client: EmbeddingClient,
    chunk_batch: ChunkBatch,
) -> EmbedWorkerResult {
    let path = chunk_batch.path;
    let content_hash = chunk_batch.content_hash;
    let chars: usize = chunk_batch
        .chunks
        .iter()
        .map(|chunk| chunk.content.chars().count())
        .sum();
    let texts: Vec<String> = chunk_batch
        .chunks
        .iter()
        .map(|chunk| chunk.content.clone())
        .collect();

    let start = Instant::now();
    match client.embed_batch(&texts).await {
        Ok(embeddings) => {
            let count = embeddings.len();
            let chunks = chunk_batch
                .chunks
                .into_iter()
                .zip(embeddings)
                .map(|(chunk, embedding)| EmbeddedChunk { chunk, embedding })
                .collect();

            EmbedWorkerResult {
                path,
                content_hash,
                count,
                chars,
                elapsed: start.elapsed(),
                chunks: Some(chunks),
                error: None,
            }
        }
        Err(err) => EmbedWorkerResult {
            path,
            content_hash,
            count: chunk_batch.chunks.len(),
            chars,
            elapsed: start.elapsed(),
            chunks: None,
            error: Some(err.to_string()),
        },
    }
}

async fn handle_embed_result(
    result: EmbedWorkerResult,
    tx: &mpsc::Sender<EmbeddedFile>,
    observer: &PipelineObserver,
    governor: Option<&mut PipelineGovernor>,
) -> bool {
    if let Some(error_message) = result.error {
        error!(
            "Embedding batch failed for {:?}: {}",
            result.path, error_message
        );
        observer
            .emit(PipelineEvent::Error {
                path: Some(result.path.clone()),
                stage: "embedder",
                message: error_message.clone(),
            })
            .await;

        if let Some(governor) = governor {
            let snapshot = observer.snapshot().await;
            if let Some(adjustment) = governor.on_error(&snapshot, &error_message) {
                observer
                    .emit(PipelineEvent::GovernorAdjusted {
                        batch_items_limit: adjustment.settings.max_batch_items,
                        batch_chars_limit: adjustment.settings.max_batch_chars,
                        concurrency_limit: adjustment.settings.concurrency,
                        mode: adjustment.mode,
                        reason: adjustment.reason,
                    })
                    .await;
            }
        }
        return true;
    }

    let Some(chunks) = result.chunks else {
        return true;
    };

    if tx
        .send(EmbeddedFile {
            path: result.path.clone(),
            content_hash: result.content_hash.clone(),
            chunks,
        })
        .await
        .is_err()
    {
        debug!("Embedder: channel closed, stopping");
        observer
            .emit(PipelineEvent::Error {
                path: Some(result.path),
                stage: "embedder",
                message: "storage channel closed".to_string(),
            })
            .await;
        return false;
    }

    observer
        .emit(PipelineEvent::ChunksEmbedded {
            path: result.path.clone(),
            content_hash: result.content_hash.clone(),
            count: result.count,
            chars: result.chars,
            elapsed: result.elapsed,
        })
        .await;

    if let Some(governor) = governor {
        let snapshot = observer.snapshot().await;
        if let Some(adjustment) = governor.on_success(result.elapsed, &snapshot) {
            observer
                .emit(PipelineEvent::GovernorAdjusted {
                    batch_items_limit: adjustment.settings.max_batch_items,
                    batch_chars_limit: adjustment.settings.max_batch_chars,
                    concurrency_limit: adjustment.settings.concurrency,
                    mode: adjustment.mode,
                    reason: adjustment.reason,
                })
                .await;
        }
    }

    true
}

// =============================================================================
// STAGE 4: STORAGE
// =============================================================================

/// Stage 4: Store embedded chunks to the database.
async fn stage_store_chunks(
    mut rx: mpsc::Receiver<EmbeddedFile>,
    storage: Arc<StorageManager>,
    observer: PipelineObserver,
) {
    while let Some(mut embedded_file) = rx.recv().await {
        let path = embedded_file.path.clone();
        let content_hash = embedded_file.content_hash.clone();
        let mut stored_for_file = 0usize;
        let mut storage_failed = false;

        while !embedded_file.chunks.is_empty() {
            let take = embedded_file.chunks.len().min(STORAGE_BATCH_SIZE);
            let batch: Vec<EmbeddedChunk> = embedded_file.chunks.drain(..take).collect();

            match store_batch(&storage, batch).await {
                Ok(count) => stored_for_file += count,
                Err(err) => {
                    error!("Storage batch failed for {:?}: {}", path, err);
                    observer
                        .emit(PipelineEvent::Error {
                            path: Some(path.clone()),
                            stage: "storage",
                            message: err.to_string(),
                        })
                        .await;
                    storage_failed = true;
                    break;
                }
            }
        }

        if !storage_failed {
            observer
                .emit(PipelineEvent::FileCommitted {
                    path,
                    content_hash,
                    chunk_count: stored_for_file,
                })
                .await;
        }
    }

    info!("Storage stage complete");
}

/// Store a batch of embedded chunks.
async fn store_batch(storage: &StorageManager, batch: Vec<EmbeddedChunk>) -> Result<usize> {
    let count = batch.len();

    let documents: Vec<ChromaDocument> = batch
        .into_iter()
        .map(|embedded| {
            if embedded.chunk.layer > 0 {
                ChromaDocument {
                    id: embedded.chunk.id,
                    namespace: embedded.chunk.namespace,
                    embedding: embedded.embedding,
                    metadata: embedded.chunk.metadata,
                    document: embedded.chunk.content,
                    layer: embedded.chunk.layer,
                    parent_id: embedded.chunk.parent_id,
                    children_ids: embedded.chunk.children_ids,
                    keywords: embedded.chunk.keywords,
                    content_hash: Some(embedded.chunk.source_hash),
                }
            } else {
                ChromaDocument::new_flat_with_hash(
                    embedded.chunk.id,
                    embedded.chunk.namespace,
                    embedded.embedding,
                    embedded.chunk.metadata,
                    embedded.chunk.content,
                    embedded.chunk.source_hash,
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

    let base_client = {
        let guard = client.lock().await;
        guard.clone()
    };
    let initial_runtime = config
        .governor
        .as_ref()
        .map(PipelineGovernorConfig::initial_settings)
        .unwrap_or_else(|| {
            let (max_batch_chars, max_batch_items) = base_client.batch_limits();
            EmbedRuntimeSettings::new(max_batch_chars, max_batch_items, config.embed_concurrency)
        });
    let (governor_mode, governor_reason) = if config.governor.is_some() {
        (
            "adaptive".to_string(),
            "warming up from conservative limits".to_string(),
        )
    } else {
        (
            "fixed".to_string(),
            "operator-configured limits".to_string(),
        )
    };
    let observer = PipelineObserver::new(
        total_files,
        config.event_sender.clone(),
        initial_runtime,
        governor_mode,
        governor_reason,
    );
    observer.emit_initial_snapshot().await;

    let (tx1, rx1) = mpsc::channel::<FileContent>(config.reader_buffer);
    let (tx2, rx2) = mpsc::channel::<ChunkBatch>(config.chunker_buffer);
    let (tx3, rx3) = mpsc::channel::<EmbeddedFile>(config.embedder_buffer);

    let storage_for_reader = storage.clone();
    let storage_for_storage = storage;
    let ns_for_reader = namespace.clone();
    let slice_mode = config.slice_mode;
    let dedup_enabled = config.dedup_enabled;

    let reader_handle = tokio::spawn(stage_read_files(
        files,
        ns_for_reader,
        storage_for_reader,
        dedup_enabled,
        tx1,
        observer.clone(),
    ));

    let chunker_handle = tokio::spawn(stage_chunk_content(rx1, tx2, slice_mode, observer.clone()));
    let embedder_handle = tokio::spawn(stage_embed_chunks(
        rx2,
        tx3,
        base_client,
        config.embed_concurrency.max(1),
        config.governor.clone(),
        observer.clone(),
    ));
    let storage_handle = tokio::spawn(stage_store_chunks(
        rx3,
        storage_for_storage,
        observer.clone(),
    ));

    let (_reader_result, _chunker_result, _embedder_result, _storage_result) = tokio::try_join!(
        reader_handle,
        chunker_handle,
        embedder_handle,
        storage_handle
    )?;

    let result = observer.result().await;

    info!(
        "Pipeline complete: {} files -> {} chunks -> {} stored",
        result.stats.files_committed, result.stats.chunks_created, result.stats.chunks_stored
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        assert_eq!(chunks[0].len(), 10);
        assert!(chunks[0].ends_with(&chunks[1][..3]));
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.reader_buffer, CHANNEL_BUFFER_SIZE);
        assert_eq!(config.slice_mode, SliceMode::default());
        assert!(config.dedup_enabled);
        assert_eq!(config.embed_concurrency, 1);
        assert!(config.governor.is_none());
        assert!(config.event_sender.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_observer_tracks_snapshot_and_failures() {
        let observer = PipelineObserver::new(
            3,
            None,
            EmbedRuntimeSettings::new(8_192, 16, 2),
            "fixed".to_string(),
            "operator-configured limits".to_string(),
        );
        observer.emit_initial_snapshot().await;

        let path_a = PathBuf::from("a.md");
        let path_b = PathBuf::from("b.md");

        observer
            .emit(PipelineEvent::FileRead {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                bytes: 10,
            })
            .await;
        observer
            .emit(PipelineEvent::ChunksCreated {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                count: 4,
            })
            .await;
        observer
            .emit(PipelineEvent::EmbedStarted {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                count: 4,
                chars: 1200,
                batch_items_limit: 16,
                batch_chars_limit: 8_192,
                concurrency_limit: 2,
            })
            .await;
        observer
            .emit(PipelineEvent::ChunksEmbedded {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                count: 4,
                chars: 1200,
                elapsed: Duration::from_millis(400),
            })
            .await;
        observer
            .emit(PipelineEvent::FileCommitted {
                path: path_a,
                content_hash: "hash-a".to_string(),
                chunk_count: 4,
            })
            .await;
        observer
            .emit(PipelineEvent::FileSkipped {
                path: path_b.clone(),
                content_hash: "hash-b".to_string(),
                reason: "duplicate".to_string(),
            })
            .await;
        observer
            .emit(PipelineEvent::Error {
                path: Some(path_b),
                stage: "embedder",
                message: "boom".to_string(),
            })
            .await;

        let result = observer.result().await;
        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.files_read, 1);
        assert_eq!(result.stats.files_skipped, 1);
        assert_eq!(result.stats.files_committed, 1);
        assert_eq!(result.stats.files_failed, 1);
        assert_eq!(result.stats.chunks_created, 4);
        assert_eq!(result.stats.chunks_embedded, 4);
        assert_eq!(result.stats.chunks_stored, 4);
        assert_eq!(result.stats.errors, 1);
        assert_eq!(result.errors.len(), 1);

        let snapshot = observer.snapshot().await;
        assert_eq!(snapshot.embed_concurrency_limit, 2);
        assert_eq!(snapshot.embed_batch_items_limit, 16);
        assert_eq!(snapshot.embed_batch_chars_limit, 8_192);
        assert!(snapshot.avg_embed_batch_ms.is_some());
        assert_eq!(snapshot.governor_mode, "fixed");
    }

    #[test]
    fn test_pipeline_governor_scales_up_only_when_backlog_stays_fast() {
        let config = PipelineGovernorConfig::adaptive(64_000, 32, 4);
        let initial_items = config.min_batch_items;
        let initial_chars = config.min_batch_chars;
        let mut governor = PipelineGovernor::new(config);

        governor.last_growth_at = Some(Instant::now() - governor.config.growth_cooldown);
        let snapshot = PipelineSnapshot {
            reader_queue_depth: 1,
            chunker_queue_depth: 4,
            storage_queue_depth: 0,
            ..Default::default()
        };

        let first = governor
            .on_success(Duration::from_millis(320), &snapshot)
            .expect("first growth adjustment");
        assert!(first.settings.max_batch_items > initial_items);

        governor.last_growth_at = Some(Instant::now() - governor.config.growth_cooldown);
        governor.current.max_batch_items = governor.config.max_batch_items;
        let second = governor
            .on_success(Duration::from_millis(320), &snapshot)
            .expect("second growth adjustment");
        assert!(second.settings.max_batch_chars > initial_chars);
    }

    #[test]
    fn test_pipeline_governor_throttles_quickly_on_pressure() {
        let config = PipelineGovernorConfig::adaptive(96_000, 48, 3);
        let mut governor = PipelineGovernor::new(config.clone());
        governor.current =
            EmbedRuntimeSettings::new(config.max_batch_chars, config.max_batch_items, 3);

        let snapshot = PipelineSnapshot {
            reader_queue_depth: 2,
            chunker_queue_depth: 5,
            storage_queue_depth: 4,
            ..Default::default()
        };

        let adjustment = governor
            .on_success(
                config.pressure_latency + Duration::from_millis(10),
                &snapshot,
            )
            .expect("pressure adjustment");
        assert!(adjustment.settings.max_batch_items < config.max_batch_items);
        assert!(adjustment.settings.max_batch_chars < config.max_batch_chars);
        assert!(adjustment.settings.concurrency < 3);
        assert!(adjustment.reason.contains("pressure"));
    }

    #[test]
    fn test_snapshot_to_stats_carries_runtime_truth() {
        let snapshot = PipelineSnapshot {
            total_files: 5,
            files_read: 4,
            files_skipped: 1,
            files_committed: 2,
            files_failed: 1,
            chunks_created: 12,
            chunks_embedded: 10,
            chunks_stored: 8,
            errors: 3,
            ..Default::default()
        };

        let stats = snapshot.to_stats();
        assert_eq!(stats.total_files, 5);
        assert_eq!(stats.files_read, 4);
        assert_eq!(stats.files_skipped, 1);
        assert_eq!(stats.files_committed, 2);
        assert_eq!(stats.files_failed, 1);
        assert_eq!(stats.chunks_created, 12);
        assert_eq!(stats.chunks_embedded, 10);
        assert_eq!(stats.chunks_stored, 8);
        assert_eq!(stats.errors, 3);
    }
}
