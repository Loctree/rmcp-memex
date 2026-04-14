use anyhow::Result;
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore, mpsc};

use rmcp_memex::{
    BM25Config, BM25Index, CrossStoreRecoveryReport, EmbeddingClient, EmbeddingConfig,
    IndexProgressTracker, PipelineConfig, PipelineEvent, PipelineSnapshot, PreprocessingConfig,
    RAGPipeline, SliceMode, StorageManager, inspect_cross_store_recovery, path_utils,
    rag::PipelineGovernorConfig, repair_cross_store_recovery,
};

#[allow(dead_code)]
fn parse_features(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Standard config discovery locations (in priority order)
#[allow(dead_code)]
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "~/.rmcp-servers/rmcp-memex/config.toml",
    "~/.config/rmcp-memex/config.toml",
    "~/.rmcp_servers/rmcp_memex/config.toml", // legacy underscore path
];

/// Discover config file from standard locations
#[allow(dead_code)]
fn discover_config() -> Option<String> {
    // 1. Environment variable takes priority
    if let Ok(path) = std::env::var("RMCP_MEMEX_CONFIG") {
        let expanded = shellexpand::tilde(&path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path);
        }
    }

    // 2. Check standard locations
    for path in CONFIG_SEARCH_PATHS {
        let expanded = shellexpand::tilde(path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path.to_string());
        }
    }

    None
}

#[allow(dead_code)]
fn load_file_config(path: &str) -> Result<FileConfig> {
    let (_canonical, contents) = path_utils::safe_read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot load config '{}': {}", path, e))?;
    toml::from_str(&contents).map_err(Into::into)
}

/// Load config from explicit path or discover from standard locations
#[allow(dead_code)]
fn load_or_discover_config(explicit_path: Option<&str>) -> Result<(FileConfig, Option<String>)> {
    // Explicit path takes priority
    if let Some(path) = explicit_path {
        return Ok((load_file_config(path)?, Some(path.to_string())));
    }

    // Try to discover config
    if let Some(discovered) = discover_config() {
        return Ok((load_file_config(&discovered)?, Some(discovered)));
    }

    // No config found - use defaults
    Ok((FileConfig::default(), None))
}

use crate::cli::config::*;
use crate::cli::definition::*;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct IndexCheckpointStats {
    pub total_files: usize,
    pub files_read: usize,
    pub files_skipped: usize,
    pub files_committed: usize,
    pub files_failed: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub chunks_stored: usize,
    pub errors: usize,
}

impl From<&PipelineSnapshot> for IndexCheckpointStats {
    fn from(snapshot: &PipelineSnapshot) -> Self {
        Self {
            total_files: snapshot.total_files,
            files_read: snapshot.files_read,
            files_skipped: snapshot.files_skipped,
            files_committed: snapshot.files_committed,
            files_failed: snapshot.files_failed,
            chunks_created: snapshot.chunks_created,
            chunks_embedded: snapshot.chunks_embedded,
            chunks_stored: snapshot.chunks_stored,
            errors: snapshot.errors,
        }
    }
}

/// Checkpoint for resumable indexing
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexCheckpoint {
    /// Namespace being indexed
    pub namespace: String,
    /// Database path tied to this checkpoint
    #[serde(default)]
    pub db_path: Option<String>,
    /// Files that have been successfully indexed (canonical paths)
    pub indexed_files: HashSet<String>,
    /// Content hashes that were durably committed or already satisfied
    #[serde(default)]
    pub indexed_hashes: HashSet<String>,
    /// When checkpoint was last updated
    pub updated_at: String,
    /// Optional runtime stats snapshot
    #[serde(default)]
    pub stats: Option<IndexCheckpointStats>,
}

impl IndexCheckpoint {
    pub fn new(namespace: &str, db_path: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            db_path: Some(db_path.to_string()),
            indexed_files: HashSet::new(),
            indexed_hashes: HashSet::new(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            stats: None,
        }
    }

    pub fn checkpoint_path(db_path: &str, namespace: &str) -> PathBuf {
        let expanded = shellexpand::tilde(db_path).to_string();
        Path::new(&expanded)
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(".index-checkpoint-{}.json", namespace))
    }

    pub fn load(db_path: &str, namespace: &str) -> Option<Self> {
        let path = Self::checkpoint_path(db_path, namespace);
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        } else {
            None
        }
    }

    pub fn save(&mut self, db_path: &str) -> Result<()> {
        self.db_path = Some(db_path.to_string());
        self.updated_at = chrono::Utc::now().to_rfc3339();
        let path = Self::checkpoint_path(db_path, &self.namespace);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn delete(db_path: &str, namespace: &str) {
        let path = Self::checkpoint_path(db_path, namespace);
        let _ = std::fs::remove_file(path);
    }

    pub fn mark_indexed(&mut self, file_path: &Path) {
        self.mark_indexed_with_hash(file_path, None);
    }

    pub fn mark_indexed_with_hash(&mut self, file_path: &Path, content_hash: Option<&str>) {
        self.indexed_files
            .insert(file_path.to_string_lossy().to_string());
        if let Some(content_hash) = content_hash {
            self.indexed_hashes.insert(content_hash.to_string());
        }
    }

    pub fn is_indexed(&self, file_path: &Path) -> bool {
        self.indexed_files
            .contains(&file_path.to_string_lossy().to_string())
    }

    pub fn update_from_snapshot(&mut self, snapshot: &PipelineSnapshot) {
        self.stats = Some(IndexCheckpointStats::from(snapshot));
    }
}

/// Configuration for batch indexing operation
pub struct BatchIndexConfig {
    pub path: PathBuf,
    pub namespace: Option<String>,
    pub recursive: bool,
    pub glob_pattern: Option<String>,
    pub max_depth: usize,
    pub db_path: String,
    pub preprocess: bool,
    /// Sanitize timestamps/UUIDs/session IDs (default: false = preserve for temporal queries)
    pub sanitize_metadata: bool,
    pub slice_mode: SliceMode,
    pub dedup: bool,
    pub embedding_config: EmbeddingConfig,
    /// Show progress bar with calibration-based ETA
    pub show_progress: bool,
    /// Resume from checkpoint if interrupted
    pub resume: bool,
    /// Enable async pipeline mode for concurrent stages
    pub pipeline: bool,
    /// Maximum embedding requests in flight for pipeline mode.
    pub pipeline_embed_concurrency: u8,
    /// Enable adaptive governor for pipeline embed throughput.
    pub pipeline_governor: bool,
    /// Number of files to process in parallel (1-16, ignored in pipeline mode)
    pub parallel: u8,
}

/// Result of indexing a single file (for parallel processing)
#[derive(Debug)]
pub enum FileIndexResult {
    /// File was indexed successfully
    Indexed,
    /// File was skipped (duplicate content)
    Skipped,
    /// File was skipped (already in checkpoint)
    SkippedResume,
    /// Indexing failed
    Failed,
}

fn pipeline_embed_rate(snapshot: &PipelineSnapshot) -> f64 {
    let elapsed_secs = snapshot.elapsed.as_secs_f64();
    if elapsed_secs > 0.0 {
        snapshot.chunks_embedded as f64 / elapsed_secs
    } else {
        0.0
    }
}

fn format_pipeline_status_line(snapshot: &PipelineSnapshot, total_files: usize) -> String {
    let terminal_files = snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
    let eta = snapshot
        .eta
        .map(HumanDuration)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "--".to_string());
    let avg_embed_ms = snapshot
        .avg_embed_batch_ms
        .map(|value| format!("{value:.0}ms"))
        .unwrap_or_else(|| "--".to_string());

    format!(
        "{}/{} files | read {} | committed {} skipped {} failed {} | chunks created {} embedded {} ({:.1}/s) stored {} ({:.1}/s) | eta {} | q {}/{}/{} | embed {}/{} req @ {} | batch {}/{} items {} / {} chars | gov {} ({}) | {}",
        terminal_files.min(total_files),
        total_files,
        snapshot.files_read,
        snapshot.files_committed,
        snapshot.files_skipped,
        snapshot.files_failed,
        snapshot.chunks_created,
        snapshot.chunks_embedded,
        pipeline_embed_rate(snapshot),
        snapshot.chunks_stored,
        snapshot.chunks_per_sec,
        eta,
        snapshot.reader_queue_depth,
        snapshot.chunker_queue_depth,
        snapshot.storage_queue_depth,
        snapshot.embed_active_requests,
        snapshot.embed_concurrency_limit,
        avg_embed_ms,
        snapshot.current_embed_batch_items,
        snapshot.embed_batch_items_limit,
        snapshot.current_embed_batch_chars,
        snapshot.embed_batch_chars_limit,
        snapshot.governor_mode,
        snapshot.governor_reason,
        snapshot.bottleneck
    )
}

fn format_pipeline_error_line(path: Option<&Path>, stage: &str, message: &str) -> String {
    match path {
        Some(path) => format!(
            "[pipeline:error] {} [{}] {}",
            stage,
            path.display(),
            message
        ),
        None => format!("[pipeline:error] {} {}", stage, message),
    }
}

struct PipelineProgressRenderer {
    total_files: usize,
    progress_bar: Option<ProgressBar>,
    last_line_at: Instant,
}

impl PipelineProgressRenderer {
    fn new(total_files: usize, interactive: bool) -> Self {
        let progress_bar = if interactive {
            let pb = ProgressBar::new(total_files as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files | {msg}")
                    .expect("invalid pipeline progress template")
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            total_files,
            progress_bar,
            last_line_at: Instant::now() - Duration::from_secs(5),
        }
    }

    fn println(&self, line: &str) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.println(line);
        } else {
            eprintln!("{}", line);
        }
    }

    fn render(&mut self, snapshot: &PipelineSnapshot, force: bool) {
        let terminal_files =
            snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
        let message = format_pipeline_status_line(snapshot, self.total_files);

        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.set_length(self.total_files as u64);
            progress_bar.set_position(terminal_files.min(self.total_files) as u64);
            progress_bar.set_message(message.clone());
            if force && terminal_files >= self.total_files {
                progress_bar.finish_with_message(format!("complete | {}", message));
            }
            return;
        }

        if !force && self.last_line_at.elapsed() < Duration::from_secs(1) {
            return;
        }

        self.last_line_at = Instant::now();
        eprintln!("[pipeline] {}", message);
    }
}

async fn consume_pipeline_events(
    mut rx: mpsc::UnboundedReceiver<PipelineEvent>,
    checkpoint: Option<Arc<Mutex<IndexCheckpoint>>>,
    db_path: String,
    show_progress: bool,
    interactive_progress: bool,
    total_files: usize,
) -> PipelineSnapshot {
    let mut renderer =
        show_progress.then(|| PipelineProgressRenderer::new(total_files, interactive_progress));
    let mut latest_snapshot = PipelineSnapshot {
        total_files,
        ..Default::default()
    };

    while let Some(event) = rx.recv().await {
        match event {
            PipelineEvent::FileCommitted {
                path, content_hash, ..
            } => {
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.mark_indexed_with_hash(&path, Some(&content_hash));
                    let _ = checkpoint.save(&db_path);
                }
            }
            PipelineEvent::FileSkipped {
                path, content_hash, ..
            } => {
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.mark_indexed_with_hash(&path, Some(&content_hash));
                    let _ = checkpoint.save(&db_path);
                }
            }
            PipelineEvent::Snapshot(snapshot) => {
                latest_snapshot = snapshot;
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.update_from_snapshot(&latest_snapshot);
                    let _ = checkpoint.save(&db_path);
                }
                if let Some(renderer) = &mut renderer {
                    renderer.render(&latest_snapshot, false);
                }
            }
            PipelineEvent::Error {
                path,
                stage,
                message,
            } => {
                if let Some(renderer) = &renderer {
                    renderer.println(&format_pipeline_error_line(
                        path.as_deref(),
                        stage,
                        &message,
                    ));
                } else {
                    eprintln!(
                        "{}",
                        format_pipeline_error_line(path.as_deref(), stage, &message)
                    );
                }
            }
            PipelineEvent::FileRead { .. }
            | PipelineEvent::EmbedStarted { .. }
            | PipelineEvent::ChunksCreated { .. }
            | PipelineEvent::ChunksEmbedded { .. }
            | PipelineEvent::GovernorAdjusted { .. } => {}
        }
    }

    if let Some(renderer) = &mut renderer {
        renderer.render(&latest_snapshot, true);
    }

    latest_snapshot
}

/// Run batch indexing with optional pipeline mode for concurrent processing
pub async fn run_batch_index(config: BatchIndexConfig) -> Result<()> {
    let BatchIndexConfig {
        path,
        namespace,
        recursive,
        glob_pattern,
        max_depth,
        db_path,
        preprocess,
        sanitize_metadata,
        slice_mode,
        dedup,
        embedding_config,
        show_progress,
        resume,
        pipeline,
        pipeline_embed_concurrency,
        pipeline_governor,
        parallel,
    } = config;
    // Expand and canonicalize path - canonicalize validates path exists and resolves symlinks
    let expanded = shellexpand::tilde(path.to_str().unwrap_or("")).to_string();
    let canonical = Path::new(&expanded).canonicalize()?;

    // Collect files
    let files = collect_files(&canonical, recursive, glob_pattern.as_deref(), max_depth)?;
    let total = files.len();

    if total == 0 {
        eprintln!("No files found matching criteria");
        return Ok(());
    }

    let mode_name = match slice_mode {
        SliceMode::Onion => "onion (hierarchical, 4 layers)",
        SliceMode::OnionFast => "onion-fast (outer+core, 2 layers)",
        SliceMode::Flat => "flat (traditional chunks)",
    };

    let use_progress_bar = show_progress && std::io::stderr().is_terminal();
    if show_progress && !use_progress_bar {
        eprintln!("Warning: --progress requires an interactive terminal (using line logs)");
    }

    let tracker = if use_progress_bar {
        let t = IndexProgressTracker::pre_scan(&files);
        t.display_pre_scan();
        Some(t)
    } else {
        eprintln!("Found {} files to index (slice mode: {})", total, mode_name);
        if preprocess {
            eprintln!("Preprocessing enabled: filtering tool artifacts, CLI output, and metadata");
        }
        if dedup {
            eprintln!("Deduplication enabled: skipping files with identical content");
        }
        None
    };

    // Initialize RAG pipeline - db_path is from CLI args or config, validated at load time
    let expanded_db = shellexpand::tilde(&db_path).to_string();
    let db_dir = Path::new(&expanded_db);
    if let Some(parent) = db_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use full storage for CLI batch indexing to ensure BM25 is written
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));
    let storage = Arc::new(StorageManager::new(&expanded_db).await?);

    let ns_name = namespace.as_deref().unwrap_or("rag");

    // Pipeline mode: concurrent stages with channels
    if pipeline {
        if preprocess {
            eprintln!("Warning: --preprocess is not supported in pipeline mode (ignoring)");
        }

        let checkpoint = if resume {
            if let Some(cp) = IndexCheckpoint::load(&db_path, ns_name) {
                let resumed_count = cp.indexed_files.len();
                eprintln!(
                    "Resuming from checkpoint: {} files already committed",
                    resumed_count
                );
                Arc::new(Mutex::new(cp))
            } else {
                Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
            }
        } else {
            IndexCheckpoint::delete(&db_path, ns_name);
            Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
        };

        let (pipeline_files, resumed_count, disable_storage_dedup) = if resume {
            let checkpoint_guard = checkpoint.lock().await;
            let resumed_count = checkpoint_guard.indexed_files.len();
            let filtered_files: Vec<PathBuf> = files
                .iter()
                .filter(|path| !checkpoint_guard.is_indexed(path))
                .cloned()
                .collect();
            let disable_storage_dedup = resumed_count > 0;
            (filtered_files, resumed_count, disable_storage_dedup)
        } else {
            (files.clone(), 0, false)
        };

        if pipeline_files.is_empty() {
            eprintln!("Pipeline resume complete: all files already committed");
            if resume {
                IndexCheckpoint::delete(&db_path, ns_name);
            }
            return Ok(());
        }

        if disable_storage_dedup {
            eprintln!(
                "Pipeline resume: using checkpoint truth for committed files to avoid partial-write false positives"
            );
        }

        eprintln!(
            "Pipeline mode: {} files ({} discovered, {} resumed), slice mode: {:?}, embed concurrency ceiling: {}, governor: {}",
            pipeline_files.len(),
            total,
            resumed_count,
            slice_mode,
            pipeline_embed_concurrency,
            if pipeline_governor {
                "adaptive"
            } else {
                "fixed"
            }
        );
        eprintln!("Running concurrent stages: reader -> chunker -> embedder -> storage");

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let event_task = tokio::spawn(consume_pipeline_events(
            event_rx,
            resume.then_some(Arc::clone(&checkpoint)),
            db_path.clone(),
            show_progress,
            use_progress_bar,
            pipeline_files.len(),
        ));

        let pipeline_config = PipelineConfig {
            slice_mode,
            dedup_enabled: dedup && !disable_storage_dedup,
            embed_concurrency: pipeline_embed_concurrency as usize,
            governor: pipeline_governor.then(|| {
                PipelineGovernorConfig::adaptive(
                    embedding_config.max_batch_chars,
                    embedding_config.max_batch_items,
                    pipeline_embed_concurrency as usize,
                )
            }),
            event_sender: Some(event_tx),
            ..Default::default()
        };

        let pipeline_run = rmcp_memex::run_pipeline(
            pipeline_files,
            ns_name.to_string(),
            storage,
            embedding_client,
            pipeline_config,
        )
        .await;
        let snapshot = event_task.await?;
        let result = pipeline_run?;

        eprintln!();
        eprintln!("Pipeline complete:");
        eprintln!("  Files committed:   {}", result.stats.files_committed);
        eprintln!("  Files read:        {}", result.stats.files_read);
        if result.stats.files_skipped > 0 {
            eprintln!("  Files skipped:     {}", result.stats.files_skipped);
        }
        if result.stats.files_failed > 0 {
            eprintln!("  Files failed:      {}", result.stats.files_failed);
        }
        if resumed_count > 0 {
            eprintln!("  Skipped (resumed): {}", resumed_count);
        }
        eprintln!("  Chunks created:    {}", result.stats.chunks_created);
        eprintln!("  Chunks embedded:   {}", result.stats.chunks_embedded);
        eprintln!("  Chunks stored:     {}", result.stats.chunks_stored);
        if result.stats.errors > 0 {
            eprintln!("  Errors:            {}", result.stats.errors);
            for error in result.errors.iter().take(5) {
                eprintln!("    - {}", error);
            }
            if result.errors.len() > 5 {
                eprintln!("    - ... and {} more", result.errors.len() - 5);
            }
        }
        if let Some(avg_embed_ms) = snapshot.avg_embed_batch_ms {
            eprintln!("  Avg embed batch:   {:.0} ms", avg_embed_ms);
        }
        eprintln!(
            "  Embed ceiling:     {} req | {} items | {} chars",
            snapshot.embed_concurrency_limit,
            snapshot.embed_batch_items_limit,
            snapshot.embed_batch_chars_limit
        );
        eprintln!(
            "  Governor:          {} ({})",
            snapshot.governor_mode, snapshot.governor_reason
        );
        eprintln!("  Bottleneck:        {}", snapshot.bottleneck);
        eprintln!("  Namespace:         {}", ns_name);
        eprintln!("  DB path:           {}", expanded_db);

        if resume && result.stats.files_failed == 0 {
            IndexCheckpoint::delete(&db_path, ns_name);
            eprintln!("Checkpoint cleared (pipeline completed successfully)");
        } else if resume && result.stats.files_failed > 0 {
            eprintln!(
                "Checkpoint preserved ({} files failed - rerun with --resume to retry)",
                result.stats.files_failed
            );
        }

        return Ok(());
    }

    // Standard (non-pipeline) mode with parallel file processing
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    // Note: preprocessing currently uses flat mode
    let effective_mode = if preprocess {
        SliceMode::Flat
    } else {
        slice_mode
    };

    // Initialize checkpoint for resume capability (wrapped for thread-safe access)
    let checkpoint = if resume {
        if let Some(cp) = IndexCheckpoint::load(&db_path, ns_name) {
            let resumed_count = cp.indexed_files.len();
            eprintln!(
                "Resuming from checkpoint: {} files already indexed",
                resumed_count
            );
            Arc::new(Mutex::new(cp))
        } else {
            Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
        }
    } else {
        // Clean start - remove any stale checkpoint
        IndexCheckpoint::delete(&db_path, ns_name);
        Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
    };

    // Atomic counters for thread-safe progress tracking
    let indexed_count = Arc::new(AtomicUsize::new(0));
    let skipped_count = Arc::new(AtomicUsize::new(0));
    let skipped_resume_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let total_chunks_count = Arc::new(AtomicUsize::new(0));
    let processed_count = Arc::new(AtomicUsize::new(0));

    // Semaphore to limit concurrent file processing
    let semaphore = Arc::new(Semaphore::new(parallel as usize));

    // Get embedder model name for calibration display
    let embedder_model = embedding_config
        .providers
        .first()
        .map(|p| p.model.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // Flag to track if calibration is complete (for progress bar)
    let calibration_done = Arc::new(AtomicBool::new(false));

    // Wrap tracker for shared access
    let tracker = tracker.map(|t| Arc::new(Mutex::new(t)));

    // Start calibration if progress mode
    if let Some(ref t) = tracker {
        t.lock().await.start_calibration();
    }

    // Create task handles for parallel processing
    let mut handles = Vec::with_capacity(files.len());

    for file_path in files.into_iter() {
        // Clone shared resources for this task
        let semaphore = Arc::clone(&semaphore);
        let rag = Arc::clone(&rag);
        let checkpoint = Arc::clone(&checkpoint);
        let tracker = tracker.clone();
        let indexed_count = Arc::clone(&indexed_count);
        let skipped_count = Arc::clone(&skipped_count);
        let skipped_resume_count = Arc::clone(&skipped_resume_count);
        let failed_count = Arc::clone(&failed_count);
        let total_chunks_count = Arc::clone(&total_chunks_count);
        let processed_count = Arc::clone(&processed_count);
        let calibration_done = Arc::clone(&calibration_done);
        let db_path = db_path.clone();
        let ns = namespace.clone();
        let canonical = canonical.clone();
        let embedder_model = embedder_model.clone();
        let _ns_name = ns_name.to_string();

        let handle = tokio::spawn(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = semaphore.acquire().await.expect("semaphore closed");

            let display_path = file_path
                .strip_prefix(&canonical)
                .unwrap_or(&file_path)
                .display()
                .to_string();

            // Check if file already indexed (resume mode)
            if resume {
                let cp = checkpoint.lock().await;
                if cp.is_indexed(&file_path) {
                    drop(cp);
                    skipped_resume_count.fetch_add(1, Ordering::SeqCst);
                    processed_count.fetch_add(1, Ordering::SeqCst);
                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    }
                    return FileIndexResult::SkippedResume;
                }
            }

            // Get file size for calibration
            let file_bytes = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            // Update progress display
            let current_processed = processed_count.load(Ordering::SeqCst);
            if let Some(ref t) = tracker {
                t.lock().await.set_message(&display_path);
            } else {
                let progress = format!("[{}/{}]", current_processed + 1, total);
                eprintln!("{} Indexing {}... ", progress, display_path);
            }

            // Build preprocessing config
            let preprocess_config = PreprocessingConfig {
                remove_metadata: sanitize_metadata,
                ..Default::default()
            };

            let result = if dedup {
                // Use dedup-enabled indexing
                if preprocess {
                    rag.index_document_with_preprocessing_and_dedup(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                } else {
                    rag.index_document_with_dedup(&file_path, ns.as_deref(), effective_mode)
                        .await
                }
            } else {
                // Use original indexing without dedup (convert to IndexResult-like outcome)
                if preprocess {
                    rag.index_document_with_preprocessing(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                    .map(|()| rmcp_memex::IndexResult::Indexed {
                        chunks_indexed: (file_bytes as usize / 500).max(1),
                        content_hash: String::new(),
                    })
                } else {
                    rag.index_document_with_mode(&file_path, ns.as_deref(), effective_mode)
                        .await
                        .map(|()| rmcp_memex::IndexResult::Indexed {
                            chunks_indexed: (file_bytes as usize / 500).max(1),
                            content_hash: String::new(),
                        })
                }
            };

            let file_result = match result {
                Ok(rmcp_memex::IndexResult::Indexed { chunks_indexed, .. }) => {
                    // Handle calibration on first completed file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(chunks_indexed, &embedder_model);
                        guard.adjust_estimate(file_bytes, chunks_indexed);
                        guard.start_progress_bar();
                    }

                    indexed_count.fetch_add(1, Ordering::SeqCst);
                    total_chunks_count.fetch_add(chunks_indexed, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_indexed(chunks_indexed);
                    } else {
                        eprintln!("  -> {} done ({} chunks)", display_path, chunks_indexed);
                    }

                    // Update checkpoint
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Indexed
                }
                Ok(rmcp_memex::IndexResult::Skipped { reason, .. }) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    skipped_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    } else {
                        eprintln!("  -> {} SKIPPED ({})", display_path, reason);
                    }

                    // Mark as indexed even if skipped (content exists)
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Skipped
                }
                Err(e) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    failed_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_failed();
                    } else {
                        eprintln!("  -> {} FAILED: {}", display_path, e);
                    }

                    FileIndexResult::Failed
                }
            };

            processed_count.fetch_add(1, Ordering::SeqCst);
            file_result
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => {
                // Task panicked - count as failure
                failed_count.fetch_add(1, Ordering::SeqCst);
                eprintln!("Task panicked: {}", e);
            }
        }
    }

    // Get final counts from atomics
    let indexed = indexed_count.load(Ordering::SeqCst);
    let skipped = skipped_count.load(Ordering::SeqCst);
    let skipped_resume = skipped_resume_count.load(Ordering::SeqCst);
    let failed = failed_count.load(Ordering::SeqCst);
    let total_chunks = total_chunks_count.load(Ordering::SeqCst);

    // Display summary
    if let Some(ref t) = tracker {
        let mut guard = t.lock().await;
        guard.finish();
        guard.display_summary();
        if skipped_resume > 0 {
            eprintln!("  Skipped (resumed): {}", skipped_resume);
        }
    } else {
        eprintln!();

        // Determine outcome and show appropriate summary
        let all_skipped = indexed == 0 && skipped > 0 && failed == 0;
        let all_failed = indexed == 0 && skipped == 0 && failed > 0;

        if all_skipped {
            eprintln!("Indexing complete: All content already indexed");
            eprintln!();
            eprintln!("  Files checked:     {}", total);
            eprintln!("  Already indexed:   {} (skipped)", skipped);
            if skipped_resume > 0 {
                eprintln!("  Resumed from:      {} (checkpoint)", skipped_resume);
            }
            eprintln!();
            eprintln!("  [OK] No new content to index - your memory is up to date!");
        } else if all_failed {
            eprintln!("Indexing FAILED: No files were indexed");
            eprintln!();
            eprintln!("  Files attempted:   {}", total);
            eprintln!("  Failed:            {}", failed);
            eprintln!();
            eprintln!("  [!] Check file permissions and embedding server connectivity");
        } else {
            eprintln!("Indexing complete:");
            eprintln!();
            eprintln!("  New chunks:        {}", total_chunks);
            eprintln!("  Files indexed:     {}", indexed);
            if dedup && skipped > 0 {
                eprintln!("  Already indexed:   {} (skipped)", skipped);
            }
            if skipped_resume > 0 {
                eprintln!("  Resumed from:      {} (checkpoint)", skipped_resume);
            }
            if failed > 0 {
                eprintln!("  Failed:            {}", failed);
            }
            eprintln!("  Total processed:   {}", total);
        }

        eprintln!();
        eprintln!("Config:");
        if let Some(ref ns) = namespace {
            eprintln!("  Namespace:         {}", ns);
        }
        eprintln!("  Slice mode:        {}", mode_name);
        eprintln!("  Parallel workers:  {}", parallel);
        eprintln!(
            "  Deduplication:     {}",
            if dedup { "enabled" } else { "disabled" }
        );
        eprintln!("  DB path:           {}", expanded_db);
    }

    // Clean up checkpoint on successful completion
    if resume && failed == 0 {
        IndexCheckpoint::delete(&db_path, ns_name);
        eprintln!("Checkpoint cleared (all files indexed successfully)");
    } else if resume && failed > 0 {
        eprintln!(
            "Checkpoint preserved ({} files failed - rerun with --resume to retry)",
            failed
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_pipeline_status_line_surfaces_stage_flow() {
        let snapshot = PipelineSnapshot {
            total_files: 6,
            files_read: 4,
            files_committed: 2,
            files_skipped: 1,
            files_failed: 1,
            chunks_created: 24,
            chunks_embedded: 20,
            chunks_stored: 18,
            errors: 0,
            files_per_sec: 0.8,
            chunks_per_sec: 3.6,
            reader_queue_depth: 2,
            chunker_queue_depth: 1,
            storage_queue_depth: 3,
            embed_active_requests: 1,
            embed_concurrency_limit: 3,
            current_embed_batch_items: 8,
            current_embed_batch_chars: 4096,
            embed_batch_items_limit: 16,
            embed_batch_chars_limit: 8192,
            avg_embed_batch_ms: Some(640.0),
            bottleneck: "storage".to_string(),
            governor_mode: "adaptive".to_string(),
            governor_reason: "backlog sustained".to_string(),
            eta: Some(Duration::from_secs(12)),
            elapsed: Duration::from_secs(5),
        };

        let line = format_pipeline_status_line(&snapshot, 6);

        assert!(line.contains("4/6 files"));
        assert!(line.contains("read 4"));
        assert!(line.contains("embedded 20 (4.0/s)"));
        assert!(line.contains("stored 18 (3.6/s)"));
        assert!(line.contains("q 2/1/3"));
        assert!(line.contains("embed 1/3 req @ 640ms"));
        assert!(line.contains("batch 8/16 items 4096 / 8192 chars"));
        assert!(line.contains("gov adaptive (backlog sustained)"));
        assert!(line.contains("storage"));
    }

    #[test]
    fn format_pipeline_error_line_keeps_stage_and_path_visible() {
        let line = format_pipeline_error_line(
            Some(Path::new("/tmp/corpus/a.md")),
            "embedder",
            "connection reset",
        );

        assert_eq!(
            line,
            "[pipeline:error] embedder [/tmp/corpus/a.md] connection reset"
        );
    }
}

fn print_cross_store_recovery_report(report: &CrossStoreRecoveryReport, execute: bool) {
    let mode = if execute { "EXECUTE" } else { "DRY RUN" };
    eprintln!("\n=== CROSS-STORE RECOVERY ({}) ===\n", mode);
    eprintln!("Recovery dir: {}", report.recovery_dir);
    eprintln!("Pending batches: {}", report.pending_batches);
    eprintln!("  Divergent:   {}", report.divergent_batches);
    eprintln!("  Rolled back: {}", report.rolled_back_batches);
    eprintln!("  Stale:       {}", report.stale_batches);
    eprintln!("  Clean:       {}", report.clean_batches);
    eprintln!("Documents examined: {}", report.documents_examined);
    eprintln!("Missing BM25 docs:  {}", report.documents_missing_bm25);
    eprintln!("Missing Lance docs: {}", report.documents_missing_lance);
    if execute {
        eprintln!("Repaired docs:      {}", report.repaired_documents);
        eprintln!("Skipped docs:       {}", report.skipped_documents);
        eprintln!("Cleared batches:    {}", report.cleared_batches);
    }

    if report.batches.is_empty() {
        eprintln!("\nNo recovery ledgers found.");
        return;
    }

    eprintln!();
    for batch in &report.batches {
        let state = match batch.state {
            rmcp_memex::CrossStoreRecoveryState::Clean => "clean",
            rmcp_memex::CrossStoreRecoveryState::Divergent => "divergent",
            rmcp_memex::CrossStoreRecoveryState::RolledBack => "rolled_back",
            rmcp_memex::CrossStoreRecoveryState::Stale => "stale",
        };
        eprintln!(
            "- {} [{}] state={} docs={} lance={} bm25={}",
            batch.batch_id,
            batch.namespace,
            state,
            batch.document_count,
            batch.lance_documents,
            batch.bm25_documents
        );
        if let Some(ref error) = batch.last_error {
            eprintln!("  last_error: {}", error);
        }
        if !batch.missing_bm25_ids.is_empty() {
            eprintln!("  missing_bm25: {}", batch.missing_bm25_ids.join(", "));
        }
        if !batch.missing_lance_ids.is_empty() {
            eprintln!("  missing_lance: {}", batch.missing_lance_ids.join(", "));
        }
    }
}

pub async fn run_repair_writes(
    db_path: String,
    namespace: Option<String>,
    execute: bool,
    json_output: bool,
) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;
    let bm25 = BM25Index::new(&BM25Config::default().with_read_only(!execute))?;

    let report = if execute {
        repair_cross_store_recovery(&storage, &bm25, namespace.as_deref()).await?
    } else {
        inspect_cross_store_recovery(&storage, &bm25, namespace.as_deref()).await?
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_cross_store_recovery_report(&report, execute);
    }

    Ok(())
}

/// Strategy for keeping documents when deduplicating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepStrategy {
    /// Keep the document with the earliest ID (lexicographic)
    Oldest,
    /// Keep the document with the latest ID (lexicographic)
    Newest,
    /// Keep the document that appears first in vector search (highest relevance)
    HighestScore,
}

impl KeepStrategy {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "newest" => Self::Newest,
            "highest-score" => Self::HighestScore,
            _ => Self::Oldest,
        }
    }
}

/// Result of deduplication operation
#[derive(Debug, Clone, Serialize)]
pub struct DedupResult {
    /// Total documents scanned
    pub total_docs: usize,
    /// Documents with unique content (no duplicates)
    pub unique_docs: usize,
    /// Duplicate groups found (each group has 2+ docs with same hash)
    pub duplicate_groups: usize,
    /// Total duplicate documents that would be/were removed
    pub duplicates_removed: usize,
    /// Documents without content_hash (cannot be deduplicated)
    pub docs_without_hash: usize,
    /// Details of each duplicate group (for reporting)
    pub groups: Vec<DedupGroup>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DedupGroup {
    pub content_hash: String,
    pub kept_id: String,
    pub kept_namespace: String,
    removed_ids: Vec<(String, String)>, // (id, namespace)
}

/// Run deduplication on the database
pub async fn run_dedup(
    namespace: Option<String>,
    dry_run: bool,
    keep_strategy: KeepStrategy,
    cross_namespace: bool,
    json_output: bool,
    db_path: String,
) -> Result<()> {
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);

    // Get all documents (optionally filtered by namespace)
    let all_docs = storage
        .all_documents(namespace.as_deref(), 1_000_000)
        .await?;

    if all_docs.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "empty",
                    "message": "No documents found",
                    "namespace": namespace,
                }))?
            );
        } else {
            eprintln!("No documents found in database.");
        }
        return Ok(());
    }

    if !json_output {
        eprintln!("Scanning {} documents for duplicates...", all_docs.len());
        if dry_run {
            eprintln!("(dry-run mode: no changes will be made)");
        }
    }

    // Group documents by content_hash
    // If cross_namespace is false, we group by (namespace, content_hash)
    // If cross_namespace is true, we group by content_hash only
    let mut hash_groups: std::collections::HashMap<String, Vec<_>> =
        std::collections::HashMap::new();
    let mut docs_without_hash = 0;

    for doc in &all_docs {
        match &doc.content_hash {
            Some(hash) if !hash.is_empty() => {
                let key = if cross_namespace {
                    hash.clone()
                } else {
                    format!("{}:{}", doc.namespace, hash)
                };
                hash_groups.entry(key).or_default().push(doc);
            }
            _ => {
                docs_without_hash += 1;
            }
        }
    }

    // Find groups with duplicates (more than 1 document per hash)
    let mut result = DedupResult {
        total_docs: all_docs.len(),
        unique_docs: 0,
        duplicate_groups: 0,
        duplicates_removed: 0,
        docs_without_hash,
        groups: Vec::new(),
    };

    for (_key, mut docs) in hash_groups {
        if docs.len() == 1 {
            result.unique_docs += 1;
            continue;
        }

        // Sort documents based on keep strategy
        match keep_strategy {
            KeepStrategy::Oldest => {
                docs.sort_by(|a, b| a.id.cmp(&b.id));
            }
            KeepStrategy::Newest => {
                docs.sort_by(|a, b| b.id.cmp(&a.id));
            }
            KeepStrategy::HighestScore => {
                // Already in search order (highest score first), no sort needed
            }
        }

        // First document is kept, rest are duplicates
        let kept = &docs[0];
        let to_remove: Vec<_> = docs[1..].to_vec();

        let group = DedupGroup {
            content_hash: kept.content_hash.clone().unwrap_or_default(),
            kept_id: kept.id.clone(),
            kept_namespace: kept.namespace.clone(),
            removed_ids: to_remove
                .iter()
                .map(|d| (d.id.clone(), d.namespace.clone()))
                .collect(),
        };

        result.duplicate_groups += 1;
        result.duplicates_removed += to_remove.len();
        result.unique_docs += 1; // The kept one is unique

        // Actually delete if not dry-run
        if !dry_run {
            for doc in &to_remove {
                storage.delete_document(&doc.namespace, &doc.id).await?;
            }
        }

        result.groups.push(group);
    }

    // Output results
    if json_output {
        let output = serde_json::json!({
            "dry_run": dry_run,
            "namespace": namespace,
            "cross_namespace": cross_namespace,
            "keep_strategy": format!("{:?}", keep_strategy).to_lowercase(),
            "result": result,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!(
            "Deduplication {}:",
            if dry_run { "report" } else { "complete" }
        );
        eprintln!("  Total documents:     {}", result.total_docs);
        eprintln!("  Unique documents:    {}", result.unique_docs);
        eprintln!("  Duplicate groups:    {}", result.duplicate_groups);
        eprintln!(
            "  Duplicates {}:  {}",
            if dry_run { "found" } else { "removed" },
            result.duplicates_removed
        );
        if result.docs_without_hash > 0 {
            eprintln!(
                "  Without hash:        {} (cannot deduplicate)",
                result.docs_without_hash
            );
        }

        // Show some duplicate groups if any
        if !result.groups.is_empty() {
            eprintln!();
            let show_count = result.groups.len().min(5);
            eprintln!(
                "Sample duplicate groups ({} of {}):",
                show_count,
                result.groups.len()
            );
            for group in result.groups.iter().take(show_count) {
                eprintln!();
                eprintln!(
                    "  Hash: {}...",
                    &group.content_hash[..group.content_hash.len().min(16)]
                );
                eprintln!("  Kept: {} (ns: {})", group.kept_id, group.kept_namespace);
                for (id, ns) in &group.removed_ids {
                    eprintln!(
                        "  {} {} (ns: {})",
                        if dry_run { "Would remove:" } else { "Removed:" },
                        id,
                        ns
                    );
                }
            }
            if result.groups.len() > 5 {
                eprintln!();
                eprintln!("  ... and {} more groups", result.groups.len() - 5);
            }
        }

        if dry_run && result.duplicates_removed > 0 {
            eprintln!();
            eprintln!("To actually remove duplicates, run with: --dry-run false");
        }
    }

    Ok(())
}

/// Migration result for reporting
#[derive(Debug, Clone, Serialize)]
pub struct MigrationResult {
    pub from_namespace: String,
    pub to_namespace: String,
    pub docs_migrated: usize,
    pub docs_merged: usize,
    pub source_deleted: bool,
    pub dry_run: bool,
}

/// Migrate documents from one namespace to another
pub async fn run_migrate_namespace(
    from: String,
    to: String,
    db_path: String,
    merge: bool,
    delete_source: bool,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Edge case: same source and target
    if from == to {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "no-op",
                    "message": "Source and target namespaces are the same",
                    "namespace": from
                }))?
            );
        } else {
            eprintln!(
                "Warning: Source and target namespaces are the same ('{}').",
                from
            );
            eprintln!("No migration needed.");
        }
        return Ok(());
    }

    // Check if source namespace exists
    let source_exists = storage.namespace_exists(&from).await?;
    if !source_exists {
        let msg = format!("Source namespace '{}' does not exist or is empty", from);
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Check if target namespace exists
    let target_exists = storage.namespace_exists(&to).await?;
    if target_exists && !merge {
        let msg = format!(
            "Target namespace '{}' already exists. Use --merge to merge documents.",
            to
        );
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg,
                    "hint": "Use --merge flag to merge into existing namespace"
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Get all documents from source namespace
    let source_docs = storage.get_all_in_namespace(&from).await?;
    let source_count = source_docs.len();

    if source_count == 0 {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "no-op",
                    "message": "Source namespace is empty",
                    "namespace": from
                }))?
            );
        } else {
            eprintln!("Source namespace '{}' is empty. Nothing to migrate.", from);
        }
        return Ok(());
    }

    // Get target document count for merge reporting
    let target_count_before = if target_exists {
        storage.count_namespace(&to).await?
    } else {
        0
    };

    if dry_run {
        // Report what would happen
        let result = MigrationResult {
            from_namespace: from.clone(),
            to_namespace: to.clone(),
            docs_migrated: source_count,
            docs_merged: if target_exists {
                target_count_before
            } else {
                0
            },
            source_deleted: delete_source,
            dry_run: true,
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "dry-run",
                    "result": result,
                    "message": "No changes made"
                }))?
            );
        } else {
            eprintln!("\n-> Dry Run: Namespace Migration\n");
            eprintln!("  From:           '{}'", from);
            eprintln!("  To:             '{}'", to);
            eprintln!("  Docs to move:   {}", source_count);
            if target_exists {
                eprintln!("  Existing docs:  {} (will be merged)", target_count_before);
            }
            eprintln!(
                "  Delete source:  {}",
                if delete_source { "yes" } else { "no" }
            );
            eprintln!("\nNo changes made (dry run).");
        }
        return Ok(());
    }

    // Perform the migration
    // Create new documents with updated namespace
    let migrated_docs: Vec<rmcp_memex::ChromaDocument> = source_docs
        .into_iter()
        .map(|mut doc| {
            doc.namespace = to.clone();
            doc
        })
        .collect();

    // Insert into target namespace
    storage.add_to_store(migrated_docs).await?;

    // Delete source namespace if requested
    let source_deleted = if delete_source {
        storage.delete_namespace_documents(&from).await?;
        true
    } else {
        false
    };

    // Report results
    let result = MigrationResult {
        from_namespace: from.clone(),
        to_namespace: to.clone(),
        docs_migrated: source_count,
        docs_merged: if target_exists {
            target_count_before
        } else {
            0
        },
        source_deleted,
        dry_run: false,
    };

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "success",
                "result": result
            }))?
        );
    } else {
        eprintln!("\n-> Namespace Migration Complete\n");
        eprintln!("  From:           '{}'", from);
        eprintln!("  To:             '{}'", to);
        eprintln!("  Docs migrated:  {}", source_count);
        if target_exists {
            eprintln!("  Merged with:    {} existing docs", target_count_before);
            eprintln!(
                "  Total in '{}': {}",
                to,
                source_count + target_count_before
            );
        }
        if source_deleted {
            eprintln!("  Source '{}': deleted", from);
        } else {
            eprintln!(
                "  Source '{}': preserved (use --delete-source to remove)",
                from
            );
        }
        eprintln!("\n  DB path: {}", db_path);
    }

    Ok(())
}

/// Purge (delete) all documents in a namespace
pub async fn run_purge_namespace(
    namespace: String,
    db_path: String,
    confirm: bool,
    json_output: bool,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Check if namespace exists
    let exists = storage.namespace_exists(&namespace).await?;
    if !exists {
        let msg = format!("Namespace '{}' does not exist or is empty", namespace);
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Get count before purge
    let docs = storage.get_all_in_namespace(&namespace).await?;
    let doc_count = docs.len();

    // Confirmation prompt (unless --confirm flag)
    if !confirm && !json_output {
        eprintln!(
            "\n⚠️  WARNING: This will permanently delete {} documents from namespace '{}'",
            doc_count, namespace
        );
        eprintln!("   This action cannot be undone!\n");
        eprint!("   Type 'yes' to confirm: ");

        use std::io::{self, BufRead, Write};
        io::stderr().flush()?;
        let stdin = io::stdin();
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;

        if input.trim().to_lowercase() != "yes" {
            eprintln!("\n   Aborted. No changes made.");
            return Ok(());
        }
    }

    // Perform the purge
    let deleted = storage.delete_namespace_documents(&namespace).await?;

    // Report results
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "success",
                "namespace": namespace,
                "documents_deleted": doc_count,
                "rows_deleted": deleted
            }))?
        );
    } else {
        eprintln!("\n✓ Purged namespace '{}'", namespace);
        eprintln!("  Documents deleted: {}", doc_count);
        eprintln!("  Rows deleted: {}", deleted);
        eprintln!("  DB path: {}", db_path);
    }

    Ok(())
}

/// Statistics for merge operation
#[derive(Debug, Clone, Default, Serialize)]
pub struct MergeStats {
    /// Total documents found in sources
    pub total_docs: usize,
    /// Documents copied to target
    pub docs_copied: usize,
    /// Documents skipped (duplicates)
    pub docs_skipped: usize,
    /// Namespaces merged
    pub namespaces: HashSet<String>,
    /// Source databases processed
    pub sources_processed: usize,
    /// Errors encountered (non-fatal)
    pub errors: usize,
}

/// Merge multiple LanceDB databases into one
pub async fn run_merge(
    source_paths: Vec<PathBuf>,
    target_path: PathBuf,
    dedup: bool,
    namespace_prefix: Option<String>,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    let mut stats = MergeStats::default();

    // Validate and sanitize source paths (prevents path traversal)
    let mut validated_sources: Vec<PathBuf> = Vec::new();
    for source in &source_paths {
        let source_str = source.to_str().unwrap_or("");
        match path_utils::sanitize_existing_path(source_str) {
            Ok(validated) => validated_sources.push(validated),
            Err(e) => {
                if !json_output {
                    eprintln!("Warning: Source database invalid: {} - {}", source_str, e);
                }
                stats.errors += 1;
            }
        }
    }

    if validated_sources.is_empty() {
        return Err(anyhow::anyhow!("No valid source databases found"));
    }

    // Validate and sanitize target path (prevents path traversal)
    let target_str = target_path.to_str().unwrap_or("");
    let validated_target = path_utils::sanitize_new_path(target_str)?;

    if !json_output {
        eprintln!("\n=== RMCP-MEMEX MERGE ===\n");
        eprintln!("Sources: {} database(s)", validated_sources.len());
        for src in &validated_sources {
            eprintln!("  - {}", src.display());
        }
        eprintln!("Target:  {}", validated_target.display());
        if let Some(ref prefix) = namespace_prefix {
            eprintln!("Prefix:  {}", prefix);
        }
        eprintln!("Dedup:   {}", if dedup { "enabled" } else { "disabled" });
        if dry_run {
            eprintln!("\n[DRY RUN - no changes will be made]\n");
        }
        eprintln!();
    }

    // Open target storage (will create if not exists)
    let target_storage = if !dry_run {
        // Ensure parent directory exists for target
        if let Some(parent) = validated_target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Some(StorageManager::new_lance_only(validated_target.to_str().unwrap_or("")).await?)
    } else {
        None
    };

    // Track content hashes for deduplication (across all sources)
    let mut seen_hashes: HashSet<String> = HashSet::new();

    // If dedup is enabled and target exists, pre-populate seen_hashes from target
    if dedup
        && !dry_run
        && let Some(ref target) = target_storage
    {
        // Get all existing documents from target to extract their hashes
        if let Ok(existing_docs) = target.all_documents(None, 100000).await {
            for doc in existing_docs {
                if let Some(hash) = doc.content_hash {
                    seen_hashes.insert(hash);
                }
            }
            if !json_output && !seen_hashes.is_empty() {
                eprintln!(
                    "Found {} existing documents in target for dedup\n",
                    seen_hashes.len()
                );
            }
        }
    }

    // Process each source database
    for source_path in &validated_sources {
        if !json_output {
            eprintln!("Processing: {}", source_path.display());
        }

        // Open source database read-only
        // SAFETY: source_path was validated by path_utils::sanitize_existing_path above
        let source_path_str = source_path.to_str().unwrap_or("");
        let source_storage = match StorageManager::new_lance_only(source_path_str).await {
            Ok(s) => s,
            Err(e) => {
                if !json_output {
                    eprintln!("  Error opening source: {}", e);
                }
                stats.errors += 1;
                continue;
            }
        };

        // Get all documents from source (using zero embedding for full scan)
        let source_docs = match source_storage.all_documents(None, 100000).await {
            Ok(docs) => docs,
            Err(e) => {
                if !json_output {
                    eprintln!("  Error reading source: {}", e);
                }
                stats.errors += 1;
                continue;
            }
        };

        if source_docs.is_empty() {
            if !json_output {
                eprintln!("  (empty database)\n");
            }
            stats.sources_processed += 1;
            continue;
        }

        let source_doc_count = source_docs.len();
        stats.total_docs += source_doc_count;

        // Group by namespace for reporting
        let mut by_namespace: std::collections::HashMap<String, Vec<_>> =
            std::collections::HashMap::new();
        for doc in source_docs {
            by_namespace
                .entry(doc.namespace.clone())
                .or_default()
                .push(doc);
        }

        if !json_output {
            eprintln!(
                "  Found {} documents in {} namespace(s)",
                source_doc_count,
                by_namespace.len()
            );
        }

        // Process each namespace
        for (ns_name, docs) in by_namespace {
            // Apply namespace prefix if specified
            let target_namespace = if let Some(ref prefix) = namespace_prefix {
                format!("{}{}", prefix, ns_name)
            } else {
                ns_name.clone()
            };

            stats.namespaces.insert(target_namespace.clone());

            let mut ns_copied = 0;
            let mut ns_skipped = 0;

            // Prepare batch for insertion
            let mut batch: Vec<rmcp_memex::ChromaDocument> = Vec::new();

            for doc in docs {
                // Check for deduplication
                if dedup && let Some(ref hash) = doc.content_hash {
                    if seen_hashes.contains(hash) {
                        ns_skipped += 1;
                        stats.docs_skipped += 1;
                        continue;
                    }
                    seen_hashes.insert(hash.clone());
                }

                // Create document with new namespace
                let new_doc = rmcp_memex::ChromaDocument {
                    id: doc.id,
                    namespace: target_namespace.clone(),
                    embedding: doc.embedding,
                    metadata: doc.metadata,
                    document: doc.document,
                    layer: doc.layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                    content_hash: doc.content_hash,
                };

                batch.push(new_doc);
                ns_copied += 1;
                stats.docs_copied += 1;
            }

            // Write batch to target (unless dry run)
            if !dry_run
                && !batch.is_empty()
                && let Some(ref target) = target_storage
                && let Err(e) = target.add_to_store(batch).await
            {
                if !json_output {
                    eprintln!("    Error writing to target: {}", e);
                }
                stats.errors += 1;
            }

            if !json_output {
                let prefix_info = if namespace_prefix.is_some() {
                    format!(" -> {}", target_namespace)
                } else {
                    String::new()
                };
                if ns_skipped > 0 {
                    eprintln!(
                        "    [{}{}] {} copied, {} skipped (duplicate)",
                        ns_name, prefix_info, ns_copied, ns_skipped
                    );
                } else {
                    eprintln!("    [{}{}] {} copied", ns_name, prefix_info, ns_copied);
                }
            }
        }

        stats.sources_processed += 1;
        if !json_output {
            eprintln!();
        }
    }

    // Output final summary
    if json_output {
        let output = serde_json::json!({
            "status": if dry_run { "dry_run" } else { "completed" },
            "sources_processed": stats.sources_processed,
            "total_docs": stats.total_docs,
            "docs_copied": stats.docs_copied,
            "docs_skipped": stats.docs_skipped,
            "namespaces": stats.namespaces.iter().collect::<Vec<_>>(),
            "namespace_count": stats.namespaces.len(),
            "errors": stats.errors,
            "target": validated_target.display().to_string(),
            "dedup_enabled": dedup,
            "namespace_prefix": namespace_prefix,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!(
            "=== MERGE {} ===\n",
            if dry_run { "PREVIEW" } else { "COMPLETE" }
        );
        eprintln!("  Sources processed: {}", stats.sources_processed);
        eprintln!("  Total documents:   {}", stats.total_docs);
        eprintln!("  Documents copied:  {}", stats.docs_copied);
        if dedup && stats.docs_skipped > 0 {
            eprintln!("  Skipped (dedup):   {}", stats.docs_skipped);
        }
        eprintln!("  Namespaces:        {}", stats.namespaces.len());
        if stats.errors > 0 {
            eprintln!("  Errors:            {}", stats.errors);
        }
        eprintln!("  Target database:   {}", validated_target.display());

        if dry_run {
            eprintln!("\n[DRY RUN - run without --dry-run to apply changes]");
        }
    }

    Ok(())
}
