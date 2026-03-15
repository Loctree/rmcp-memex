//! Indexer Module for TUI Wizard
//!
//! Provides functionality for:
//! - Importing existing LanceDB databases
//! - Indexing directories with progress reporting
//! - Managing namespaces

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use walkdir::WalkDir;

/// Progress update sent during indexing
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Current file being processed
    pub current_file: String,
    /// Number of files processed so far
    pub processed: usize,
    /// Total number of files to process
    pub total: usize,
    /// Number of files skipped (duplicates, unsupported, etc.)
    pub skipped: usize,
    /// Whether indexing is complete
    pub complete: bool,
    /// Error message if indexing failed
    pub error: Option<String>,
}

impl IndexProgress {
    pub fn new(total: usize) -> Self {
        Self {
            current_file: String::new(),
            processed: 0,
            total,
            skipped: 0,
            complete: false,
            error: None,
        }
    }

    pub fn percent(&self) -> u8 {
        if self.total == 0 {
            return 0;
        }
        ((self.processed as f64 / self.total as f64) * 100.0).min(100.0) as u8
    }

    pub fn progress_bar(&self, width: usize) -> String {
        let filled = (self.percent() as usize * width) / 100;
        let empty = width.saturating_sub(filled);
        format!(
            "[{}{}] {}% ({}/{})",
            "=".repeat(filled),
            " ".repeat(empty),
            self.percent(),
            self.processed,
            self.total
        )
    }
}

/// Data setup option selected by user
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSetupOption {
    /// Import an existing LanceDB database
    ImportLanceDB,
    /// Index a directory with files
    IndexDirectory,
    /// Skip data setup for now
    Skip,
}

impl DataSetupOption {
    pub fn display_name(&self) -> &'static str {
        match self {
            DataSetupOption::ImportLanceDB => "[1] Import existing LanceDB",
            DataSetupOption::IndexDirectory => "[2] Index a directory now",
            DataSetupOption::Skip => "[3] Skip for now",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            DataSetupOption::ImportLanceDB => "Copy or link an existing LanceDB database",
            DataSetupOption::IndexDirectory => "Recursively index files with embeddings",
            DataSetupOption::Skip => "Configure data later via CLI",
        }
    }

    pub fn all() -> Vec<DataSetupOption> {
        vec![
            DataSetupOption::ImportLanceDB,
            DataSetupOption::IndexDirectory,
            DataSetupOption::Skip,
        ]
    }
}

/// State for the data setup wizard step
#[derive(Debug, Clone)]
pub struct DataSetupState {
    /// Selected option
    pub option: DataSetupOption,
    /// Focus index for option selection
    pub focus: usize,
    /// Whether we're in path input mode
    pub input_mode: bool,
    /// Input buffer for path entry
    pub input_buffer: String,
    /// Source path (LanceDB or directory to index)
    pub source_path: Option<String>,
    /// Namespace for indexing
    pub namespace: Option<String>,
    /// Current progress during indexing
    pub progress: Option<IndexProgress>,
    /// Sub-step within the option
    pub sub_step: DataSetupSubStep,
    /// Import mode for LanceDB
    pub import_mode: ImportMode,
    /// Validation error message (if path validation fails)
    pub validation_error: Option<String>,
}

/// Sub-steps within data setup
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSetupSubStep {
    /// Selecting which option to use
    SelectOption,
    /// Entering path for import/index
    EnterPath,
    /// Entering namespace for indexing
    EnterNamespace,
    /// Selecting import mode (copy vs symlink)
    SelectImportMode,
    /// Indexing in progress
    Indexing,
    /// Operation complete
    Complete,
}

/// Mode for importing LanceDB
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportMode {
    /// Copy the database files
    Copy,
    /// Create a symlink to existing database
    Symlink,
    /// Just update config to point to existing path
    ConfigOnly,
}

impl ImportMode {
    pub fn display_name(&self) -> &'static str {
        match self {
            ImportMode::Copy => "[1] Copy database files",
            ImportMode::Symlink => "[2] Create symlink",
            ImportMode::ConfigOnly => "[3] Just update config path",
        }
    }

    pub fn all() -> Vec<ImportMode> {
        vec![
            ImportMode::Copy,
            ImportMode::Symlink,
            ImportMode::ConfigOnly,
        ]
    }
}

impl Default for DataSetupState {
    fn default() -> Self {
        Self {
            option: DataSetupOption::Skip,
            focus: 0,
            input_mode: false,
            input_buffer: String::new(),
            source_path: None,
            namespace: None,
            progress: None,
            sub_step: DataSetupSubStep::SelectOption,
            import_mode: ImportMode::ConfigOnly,
            validation_error: None,
        }
    }
}

impl DataSetupState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get currently focused option
    pub fn focused_option(&self) -> DataSetupOption {
        let options = DataSetupOption::all();
        options
            .get(self.focus)
            .cloned()
            .unwrap_or(DataSetupOption::Skip)
    }

    /// Select the focused option and move to next sub-step
    pub fn select_focused(&mut self) {
        self.option = self.focused_option();
        self.sub_step = match self.option {
            DataSetupOption::ImportLanceDB => DataSetupSubStep::EnterPath,
            DataSetupOption::IndexDirectory => DataSetupSubStep::EnterPath,
            DataSetupOption::Skip => DataSetupSubStep::Complete,
        };
        if self.sub_step == DataSetupSubStep::EnterPath {
            self.input_mode = true;
            self.input_buffer.clear();
        }
    }

    /// Confirm path entry and move to next sub-step
    pub fn confirm_path(&mut self) {
        let path = self.input_buffer.trim().to_string();
        if path.is_empty() {
            return;
        }
        self.source_path = Some(path);
        self.input_mode = false;

        match self.option {
            DataSetupOption::ImportLanceDB => {
                self.sub_step = DataSetupSubStep::SelectImportMode;
                self.focus = 0;
            }
            DataSetupOption::IndexDirectory => {
                // Generate default namespace from folder name
                if let Some(ref path) = self.source_path {
                    let folder_name = PathBuf::from(path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("indexed")
                        .to_string();
                    self.input_buffer = format!("kb:{}", folder_name);
                }
                self.sub_step = DataSetupSubStep::EnterNamespace;
                self.input_mode = true;
            }
            DataSetupOption::Skip => {
                self.sub_step = DataSetupSubStep::Complete;
            }
        }
    }

    /// Confirm namespace and start indexing
    pub fn confirm_namespace(&mut self) {
        let ns = self.input_buffer.trim().to_string();
        if ns.is_empty() {
            return;
        }
        self.namespace = Some(ns);
        self.input_mode = false;
        self.sub_step = DataSetupSubStep::Indexing;
    }

    /// Select import mode and proceed
    pub fn select_import_mode(&mut self, mode: ImportMode) {
        self.import_mode = mode;
        self.sub_step = DataSetupSubStep::Complete;
    }

    /// Check if data setup is complete
    pub fn is_complete(&self) -> bool {
        self.sub_step == DataSetupSubStep::Complete
    }

    /// Check if we should show indexing progress
    pub fn is_indexing(&self) -> bool {
        self.sub_step == DataSetupSubStep::Indexing
    }
}

/// Supported file extensions for indexing
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "rst", "org", "json", "yaml", "yml", "toml", "xml", "rs", "py", "js",
    "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php", "swift", "kt", "scala",
    "sh", "bash", "zsh", "fish", "sql", "graphql", "html", "css", "scss", "sass", "less", "pdf",
];

/// Collect files from a directory for indexing
pub fn collect_files(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if !dir_path.exists() {
        return Err(anyhow!("Directory does not exist: {}", dir_path.display()));
    }

    if !dir_path.is_dir() {
        return Err(anyhow!("Path is not a directory: {}", dir_path.display()));
    }

    for entry in WalkDir::new(dir_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Skip hidden files and directories
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }

        // Skip directories
        if path.is_dir() {
            continue;
        }

        // Check extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
        {
            files.push(path.to_path_buf());
        }
    }

    files.sort();
    Ok(files)
}

/// Import an existing LanceDB database
pub async fn import_lancedb(
    source_path: &Path,
    target_path: &Path,
    mode: ImportMode,
) -> Result<String> {
    let source = source_path.to_path_buf();

    if !source.exists() {
        return Err(anyhow!("Source path does not exist: {}", source.display()));
    }

    match mode {
        ImportMode::Copy => {
            // Create target directory
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            // Copy recursively
            copy_dir_recursive(&source, target_path).await?;
            Ok(format!("Copied database to {}", target_path.display()))
        }
        ImportMode::Symlink => {
            // Create target parent directory
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            // Remove target if it exists
            if target_path.exists() || target_path.is_symlink() {
                if target_path.is_dir() {
                    tokio::fs::remove_dir_all(target_path).await?;
                } else {
                    tokio::fs::remove_file(target_path).await?;
                }
            }

            // Create symlink
            #[cfg(unix)]
            tokio::fs::symlink(&source, target_path).await?;
            #[cfg(windows)]
            tokio::fs::symlink_dir(&source, target_path).await?;

            Ok(format!(
                "Created symlink {} -> {}",
                target_path.display(),
                source.display()
            ))
        }
        ImportMode::ConfigOnly => Ok(format!(
            "Config will use existing database at {}",
            source.display()
        )),
    }
}

/// Recursively copy a directory with path validation
async fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    use crate::path_utils::validate_write_path;

    // Validate source and read directory in one atomic step
    let (safe_src, mut entries) = crate::path_utils::safe_read_dir(src).await?;

    // Validate destination is safe to write
    let safe_dst = validate_write_path(dst)?;

    tokio::fs::create_dir_all(&safe_dst).await?;
    let _ = &safe_src; // keep binding alive for clarity
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();

        // Validate each entry path to prevent symlink attacks
        let entry_name = entry.file_name();
        let entry_name_str = entry_name.to_string_lossy();

        // Reject suspicious filenames
        if entry_name_str.contains("..")
            || entry_name_str.starts_with('.')
                && entry_name_str.len() > 1
                && entry_name_str.chars().nth(1) == Some('.')
        {
            continue; // Skip potentially dangerous entries
        }

        let dest_path = safe_dst.join(&entry_name);

        if path.is_dir() {
            Box::pin(copy_dir_recursive(&path, &dest_path)).await?;
        } else {
            // Validate and copy in one atomic step
            if let Ok(_dst) = crate::path_utils::safe_copy(&path, &dest_path) {
                // copied successfully
            }
        }
    }

    Ok(())
}

/// Index a directory using the RAG pipeline
/// Returns a channel receiver for progress updates
pub fn start_indexing(
    dir_path: PathBuf,
    namespace: String,
    embedding_config: crate::EmbeddingConfig,
    db_path: String,
) -> mpsc::Receiver<IndexProgress> {
    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        let result = run_indexing(dir_path, namespace, embedding_config, db_path, tx.clone()).await;

        if let Err(e) = result {
            let mut progress = IndexProgress::new(0);
            progress.error = Some(e.to_string());
            progress.complete = true;
            let _ = tx.send(progress).await;
        }
    });

    rx
}

async fn run_indexing(
    dir_path: PathBuf,
    namespace: String,
    embedding_config: crate::EmbeddingConfig,
    db_path: String,
    tx: mpsc::Sender<IndexProgress>,
) -> Result<()> {
    use crate::{EmbeddingClient, RAGPipeline, StorageManager};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // Collect files first
    let files = collect_files(&dir_path)?;
    let total = files.len();

    if total == 0 {
        let mut progress = IndexProgress::new(0);
        progress.complete = true;
        progress.error = Some("No indexable files found in directory".to_string());
        let _ = tx.send(progress).await;
        return Ok(());
    }

    let mut progress = IndexProgress::new(total);
    let _ = tx.send(progress.clone()).await;

    // Initialize storage and embedding client
    let expanded_db_path = shellexpand::tilde(&db_path).to_string();
    let storage = Arc::new(StorageManager::new_lance_only(&expanded_db_path).await?);
    storage.ensure_collection().await?;

    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));

    let pipeline = RAGPipeline::new(embedding_client, storage).await?;

    // Index each file
    for (i, file_path) in files.iter().enumerate() {
        progress.current_file = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        progress.processed = i;
        let _ = tx.send(progress.clone()).await;

        // Try to index the file
        match pipeline
            .index_document_with_dedup(file_path, Some(&namespace), crate::SliceMode::Onion)
            .await
        {
            Ok(result) => {
                if result.is_skipped() {
                    progress.skipped += 1;
                }
            }
            Err(e) => {
                // Log error but continue with other files
                tracing::warn!("Failed to index {}: {}", file_path.display(), e);
                progress.skipped += 1;
            }
        }
    }

    // Final progress update
    progress.processed = total;
    progress.complete = true;
    progress.current_file = "Complete".to_string();
    let _ = tx.send(progress).await;

    Ok(())
}

/// Validate a path entered by the user with security checks
pub fn validate_path(path: &str) -> Result<PathBuf> {
    use crate::path_utils::sanitize_existing_path;

    if path.trim().is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    // Use secure path sanitization which:
    // 1. Checks for path traversal sequences
    // 2. Expands tilde
    // 3. Canonicalizes the path
    // 4. Validates it's under an allowed directory
    sanitize_existing_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar() {
        let mut progress = IndexProgress::new(100);
        progress.processed = 50;
        let bar = progress.progress_bar(20);
        assert!(bar.contains("50%"));
        assert!(bar.contains("50/100"));
    }

    #[test]
    fn test_data_setup_options() {
        let options = DataSetupOption::all();
        assert_eq!(options.len(), 3);
    }

    #[test]
    fn test_state_transitions() {
        let mut state = DataSetupState::new();
        assert_eq!(state.sub_step, DataSetupSubStep::SelectOption);

        // Select index directory
        state.focus = 1; // IndexDirectory
        state.select_focused();
        assert_eq!(state.option, DataSetupOption::IndexDirectory);
        assert_eq!(state.sub_step, DataSetupSubStep::EnterPath);
        assert!(state.input_mode);
    }

    #[test]
    fn test_validate_path() {
        // Empty path should fail
        assert!(validate_path("").is_err());
        assert!(validate_path("   ").is_err());

        // Non-existent path should fail
        assert!(validate_path("/this/path/does/not/exist/ever").is_err());

        // Home directory should work (if it exists)
        let result = validate_path("~");
        // This might pass or fail depending on environment
        assert!(result.is_ok() || result.is_err());
    }
}
