//! TUI Wizard Application Logic
//!
//! Main application state and step management for the configuration wizard.
//! Implements the new wizard flow with EmbedderSetup as the first configuration step.

use crate::embeddings::{EmbeddingConfig, ProviderConfig};
use crate::tui::detection::{
    DetectedProvider, ProviderKind, check_health, detect_providers, dimension_explanation,
};
use crate::tui::health::{HealthCheckResult, HealthChecker};
use crate::tui::host_detection::{
    ExtendedHostKind, HostDetection, detect_extended_hosts, generate_extended_snippet,
    write_extended_host_config,
};
use crate::tui::indexer::{
    DataSetupOption, DataSetupState, DataSetupSubStep, ImportMode, IndexProgress, import_lancedb,
    start_indexing,
};
use anyhow::Result;
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use std::io::{Stdout, stdout};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

/// Configuration for running the wizard.
#[derive(Debug, Clone, Default)]
pub struct WizardConfig {
    pub config_path: Option<String>,
    pub dry_run: bool,
}

/// Wizard step enum - new flow with EmbedderSetup first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WizardStep {
    Welcome,
    EmbedderSetup,
    MemexSettings,
    HostSelection,
    SnippetPreview,
    HealthCheck,
    DataSetup,
    Summary,
}

impl WizardStep {
    pub fn title(&self) -> &'static str {
        match self {
            WizardStep::Welcome => "Welcome",
            WizardStep::EmbedderSetup => "Embedder Setup",
            WizardStep::MemexSettings => "Database Setup",
            WizardStep::HostSelection => "Host Selection",
            WizardStep::SnippetPreview => "Config Preview",
            WizardStep::HealthCheck => "Health Check",
            WizardStep::DataSetup => "Data Setup",
            WizardStep::Summary => "Summary & Write",
        }
    }

    pub fn next(&self) -> Option<WizardStep> {
        match self {
            WizardStep::Welcome => Some(WizardStep::EmbedderSetup),
            WizardStep::EmbedderSetup => Some(WizardStep::MemexSettings),
            WizardStep::MemexSettings => Some(WizardStep::HostSelection),
            WizardStep::HostSelection => Some(WizardStep::SnippetPreview),
            WizardStep::SnippetPreview => Some(WizardStep::HealthCheck),
            WizardStep::HealthCheck => Some(WizardStep::DataSetup),
            WizardStep::DataSetup => Some(WizardStep::Summary),
            WizardStep::Summary => None,
        }
    }

    pub fn prev(&self) -> Option<WizardStep> {
        match self {
            WizardStep::Welcome => None,
            WizardStep::EmbedderSetup => Some(WizardStep::Welcome),
            WizardStep::MemexSettings => Some(WizardStep::EmbedderSetup),
            WizardStep::HostSelection => Some(WizardStep::MemexSettings),
            WizardStep::SnippetPreview => Some(WizardStep::HostSelection),
            WizardStep::HealthCheck => Some(WizardStep::SnippetPreview),
            WizardStep::DataSetup => Some(WizardStep::HealthCheck),
            WizardStep::Summary => Some(WizardStep::DataSetup),
        }
    }

    pub fn step_number(&self) -> usize {
        match self {
            WizardStep::Welcome => 1,
            WizardStep::EmbedderSetup => 2,
            WizardStep::MemexSettings => 3,
            WizardStep::HostSelection => 4,
            WizardStep::SnippetPreview => 5,
            WizardStep::HealthCheck => 6,
            WizardStep::DataSetup => 7,
            WizardStep::Summary => 8,
        }
    }

    pub fn total_steps() -> usize {
        8
    }
}

/// Embedder configuration state for the wizard.
#[derive(Debug, Clone)]
pub struct EmbedderState {
    /// Detected embedding providers from auto-detection
    pub detected_providers: Vec<DetectedProvider>,
    /// Whether detection is currently running
    pub detecting: bool,
    /// Selected provider (from detection or manual)
    pub selected_provider: Option<DetectedProvider>,
    /// Manual base URL (if configuring manually)
    pub manual_url: String,
    /// Manual model name
    pub manual_model: String,
    /// Required embedding dimension
    pub dimension: usize,
    /// Whether to use manual configuration instead of detected
    pub use_manual: bool,
}

impl Default for EmbedderState {
    fn default() -> Self {
        Self {
            detected_providers: Vec::new(),
            detecting: false,
            selected_provider: None,
            manual_url: "http://localhost:11434".to_string(),
            manual_model: "qwen3-embedding:8b".to_string(),
            dimension: 4096,
            use_manual: false,
        }
    }
}

impl EmbedderState {
    /// Get dimension explanation text
    pub fn dimension_hint(&self) -> &'static str {
        dimension_explanation(self.dimension)
    }

    /// Update embedding config from state
    pub fn to_embedding_config(&self) -> EmbeddingConfig {
        let provider = if self.use_manual {
            ProviderConfig {
                name: "manual".to_string(),
                base_url: self.manual_url.clone(),
                model: self.manual_model.clone(),
                priority: 1,
                ..Default::default()
            }
        } else if let Some(ref detected) = self.selected_provider {
            ProviderConfig {
                name: match detected.kind {
                    ProviderKind::Ollama => "ollama-local".to_string(),
                    ProviderKind::Mlx => "mlx-local".to_string(),
                    ProviderKind::OpenAICompat => "openai-compat".to_string(),
                    ProviderKind::Manual => "manual".to_string(),
                },
                base_url: detected.base_url.clone(),
                model: detected.model().unwrap_or("unknown").to_string(),
                priority: 1,
                ..Default::default()
            }
        } else {
            // Fallback default
            ProviderConfig {
                name: "ollama-local".to_string(),
                base_url: "http://localhost:11434".to_string(),
                model: "qwen3-embedding:8b".to_string(),
                priority: 1,
                ..Default::default()
            }
        };

        EmbeddingConfig {
            required_dimension: self.dimension,
            providers: vec![provider],
            ..Default::default()
        }
    }
}

/// Get current hostname (machine-agnostic)
fn get_hostname() -> String {
    // Try gethostname syscall first
    if let Some(name) = std::process::Command::new("hostname")
        .arg("-s") // short name without domain
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let hostname = String::from_utf8_lossy(&name.stdout).trim().to_string();
        if !hostname.is_empty() {
            return hostname;
        }
    }

    // Fallback to environment variables
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "local".to_string())
}

/// Database path mode for multi-host setups
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbPathMode {
    /// Single shared path (e.g., ~/.ai-memories/lancedb)
    Shared,
    /// Per-host path with hostname suffix (e.g., ~/.ai-memories/lancedb.dragon)
    PerHost,
}

/// Editable memex configuration.
#[derive(Debug, Clone)]
pub struct MemexCfg {
    pub db_path: String,
    pub cache_mb: usize,
    pub log_level: String,
    pub max_request_bytes: usize,
    pub mode: String,
    /// Current machine hostname (auto-detected)
    pub hostname: String,
    /// Database path mode (shared vs per-host)
    pub db_path_mode: DbPathMode,
    /// HTTP/SSE server port (None = disabled, Some(port) = enabled)
    pub http_port: Option<u16>,
}

impl Default for MemexCfg {
    fn default() -> Self {
        let hostname = get_hostname();
        Self {
            // New default path per requirements
            db_path: "~/.ai-memories/lancedb".to_string(),
            cache_mb: 4096,
            log_level: "info".to_string(),
            max_request_bytes: 10 * 1024 * 1024, // 10MB
            mode: "full".to_string(),
            hostname,
            db_path_mode: DbPathMode::Shared,
            http_port: None,
        }
    }
}

impl MemexCfg {
    /// Get the effective database path (with hostname suffix if per-host mode)
    pub fn effective_db_path(&self) -> String {
        match self.db_path_mode {
            DbPathMode::Shared => self.db_path.clone(),
            DbPathMode::PerHost => format!("{}.{}", self.db_path, self.hostname),
        }
    }
}

/// Main application state.
pub struct App {
    pub step: WizardStep,
    pub memex_cfg: MemexCfg,
    /// Embedder configuration state (new EmbedderSetup step)
    pub embedder_state: EmbedderState,
    /// Derived embedding config (updated from embedder_state)
    pub embedding_config: EmbeddingConfig,
    /// Extended hosts with their kind and detection info
    pub hosts: Vec<(ExtendedHostKind, HostDetection)>,
    pub selected_hosts: Vec<usize>,
    pub dry_run: bool,
    pub messages: Vec<String>,
    pub focus: usize,
    pub binary_path: String,
    pub health_status: Option<String>,
    pub should_quit: bool,
    pub input_mode: bool,
    pub input_buffer: String,
    pub editing_field: Option<usize>,
    /// Enhanced health check result
    pub health_result: Option<HealthCheckResult>,
    /// Whether health check is currently running
    pub health_running: bool,
    /// Data setup state
    pub data_setup: DataSetupState,
    /// Progress receiver for indexing operations
    pub index_progress_rx: Option<mpsc::Receiver<IndexProgress>>,
    /// Whether rmcp-memex config has been written
    pub config_written: bool,
}

impl App {
    pub fn new(config: WizardConfig) -> Self {
        let hosts = detect_extended_hosts();
        let binary_path = which_rmcp_memex().unwrap_or_else(|| "rmcp_memex".to_string());
        let embedder_state = EmbedderState::default();
        let embedding_config = embedder_state.to_embedding_config();

        Self {
            step: WizardStep::Welcome,
            memex_cfg: MemexCfg::default(),
            embedder_state,
            embedding_config,
            hosts,
            selected_hosts: Vec::new(),
            dry_run: config.dry_run,
            messages: Vec::new(),
            focus: 0,
            binary_path,
            health_status: None,
            should_quit: false,
            input_mode: false,
            input_buffer: String::new(),
            editing_field: None,
            health_result: None,
            health_running: false,
            data_setup: DataSetupState::new(),
            index_progress_rx: None,
            config_written: false,
        }
    }

    pub fn next_step(&mut self) {
        if let Some(next) = self.step.next() {
            // On leaving EmbedderSetup, update the embedding config
            if self.step == WizardStep::EmbedderSetup {
                self.embedding_config = self.embedder_state.to_embedding_config();
            }
            self.step = next;
            self.focus = 0;
            self.input_mode = false;
            self.editing_field = None;

            // Trigger actions on entering specific steps
            if self.step == WizardStep::EmbedderSetup
                && self.embedder_state.detected_providers.is_empty()
            {
                self.embedder_state.detecting = true;
            }

            // Auto-trigger health check when entering HealthCheck step
            if self.step == WizardStep::HealthCheck && !self.health_running {
                self.run_health_check();
                self.trigger_health_check();
            }
        }
    }

    pub fn prev_step(&mut self) {
        if let Some(prev) = self.step.prev() {
            self.step = prev;
            self.focus = 0;
        }
    }

    pub fn toggle_host(&mut self, idx: usize) {
        if self.selected_hosts.contains(&idx) {
            self.selected_hosts.retain(|&i| i != idx);
        } else {
            self.selected_hosts.push(idx);
        }
    }

    pub fn get_selected_hosts(&self) -> Vec<&(ExtendedHostKind, HostDetection)> {
        self.selected_hosts
            .iter()
            .filter_map(|&i| self.hosts.get(i))
            .collect()
    }

    pub fn generate_snippets(&self) -> Vec<(ExtendedHostKind, String)> {
        let effective_path = self.memex_cfg.effective_db_path();
        self.get_selected_hosts()
            .iter()
            .map(|(kind, _detection)| {
                let mut snippet =
                    generate_extended_snippet(*kind, &self.binary_path, &effective_path);
                // Add HTTP port arg if configured
                if let Some(port) = self.memex_cfg.http_port {
                    snippet = snippet.replace(
                        "\"serve\"",
                        &format!("\"serve\", \"--http-port\", \"{}\"", port),
                    );
                }
                (*kind, snippet)
            })
            .collect()
    }

    pub fn run_health_check(&mut self) {
        self.health_status = Some("Checking...".to_string());

        // Try to run rmcp_memex --version
        match std::process::Command::new(&self.binary_path)
            .arg("--version")
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    self.health_status = Some(format!("[OK] Binary OK: {}", version.trim()));
                } else {
                    self.health_status = Some("[ERR] Binary found but failed to run".to_string());
                }
            }
            Err(e) => {
                self.health_status = Some(format!("[ERR] Binary not found: {}", e));
            }
        }

        // Show hostname info
        self.messages.push(format!(
            "[INFO] Host: {} (path mode: {:?})",
            self.memex_cfg.hostname, self.memex_cfg.db_path_mode
        ));

        // Check db_path (use effective path)
        let effective_path = self.memex_cfg.effective_db_path();
        let expanded_path = shellexpand::tilde(&effective_path).to_string();
        let db_path = PathBuf::from(&expanded_path);
        if db_path.exists() {
            self.messages
                .push(format!("[OK] DB path exists: {}", expanded_path));
        } else {
            self.messages
                .push(format!("[-] DB path will be created: {}", expanded_path));
        }

        // Show HTTP port info
        if let Some(port) = self.memex_cfg.http_port {
            self.messages
                .push(format!("[INFO] HTTP/SSE server will run on port {}", port));
        }
    }

    pub fn write_configs(&mut self) -> Result<()> {
        let effective_path = self.memex_cfg.effective_db_path();

        if self.dry_run {
            self.messages.push("DRY RUN: No files written".to_string());
            self.messages.push(format!(
                "Host: {} | Path mode: {:?}",
                self.memex_cfg.hostname, self.memex_cfg.db_path_mode
            ));
            for &idx in &self.selected_hosts.clone() {
                if let Some((kind, detection)) = self.hosts.get(idx) {
                    let snippet =
                        generate_extended_snippet(*kind, &self.binary_path, &effective_path);
                    self.messages.push(format!(
                        "Would write to {} ({}):\n{}",
                        kind.display_name(),
                        detection.path.display(),
                        snippet
                    ));
                }
            }
            return Ok(());
        }

        let mut success_count = 0;
        let mut error_count = 0;

        for &idx in &self.selected_hosts.clone() {
            if let Some((kind, _detection)) = self.hosts.get(idx) {
                match write_extended_host_config(*kind, &self.binary_path, &effective_path) {
                    Ok(result) => {
                        success_count += 1;
                        if let Some(backup) = result.backup_path {
                            self.messages.push(format!(
                                "[OK] {} backup: {}",
                                result.host_name,
                                backup.display()
                            ));
                        }
                        if result.created {
                            self.messages.push(format!(
                                "[OK] {} created: {}",
                                result.host_name,
                                result.config_path.display()
                            ));
                        } else {
                            self.messages.push(format!(
                                "[OK] {} updated: {}",
                                result.host_name,
                                result.config_path.display()
                            ));
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        self.messages
                            .push(format!("[ERR] {} failed: {}", kind.display_name(), e));
                    }
                }
            }
        }

        if success_count > 0 {
            self.messages.push(format!(
                "\nConfiguration complete! {} host(s) configured.",
                success_count
            ));
        }
        if error_count > 0 {
            self.messages.push(format!(
                "Warning: {} host(s) failed to configure.",
                error_count
            ));
        }

        Ok(())
    }

    fn settings_field_count(&self) -> usize {
        7 // db_path, db_path_mode, http_port, cache_mb, log_level, max_request_bytes, mode
    }

    pub fn get_field_value(&self, field: usize) -> String {
        match field {
            0 => self.memex_cfg.db_path.clone(),
            1 => match self.memex_cfg.db_path_mode {
                DbPathMode::Shared => "shared".to_string(),
                DbPathMode::PerHost => format!("per-host ({})", self.memex_cfg.hostname),
            },
            2 => match self.memex_cfg.http_port {
                Some(port) => port.to_string(),
                None => "disabled".to_string(),
            },
            3 => self.memex_cfg.cache_mb.to_string(),
            4 => self.memex_cfg.log_level.clone(),
            5 => self.memex_cfg.max_request_bytes.to_string(),
            6 => self.memex_cfg.mode.clone(),
            _ => String::new(),
        }
    }

    pub fn set_field_value(&mut self, field: usize, value: String) {
        match field {
            0 => self.memex_cfg.db_path = value,
            1 => {
                // Toggle between shared and per-host
                self.memex_cfg.db_path_mode = match self.memex_cfg.db_path_mode {
                    DbPathMode::Shared => DbPathMode::PerHost,
                    DbPathMode::PerHost => DbPathMode::Shared,
                };
            }
            2 => {
                // Parse port or disable
                if value.to_lowercase() == "disabled" || value.is_empty() {
                    self.memex_cfg.http_port = None;
                } else if let Ok(port) = value.parse() {
                    self.memex_cfg.http_port = Some(port);
                }
            }
            3 => {
                if let Ok(v) = value.parse() {
                    self.memex_cfg.cache_mb = v;
                }
            }
            4 => self.memex_cfg.log_level = value,
            5 => {
                if let Ok(v) = value.parse() {
                    self.memex_cfg.max_request_bytes = v;
                }
            }
            6 => self.memex_cfg.mode = value,
            _ => {}
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) {
        // Handle input mode for settings and data setup
        if self.input_mode || self.data_setup.input_mode {
            self.handle_input_key(key);
            return;
        }

        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Esc => {
                if self.step != WizardStep::Welcome {
                    self.prev_step();
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Enter | KeyCode::Tab => self.handle_enter(),
            KeyCode::Right | KeyCode::Char('n') => self.handle_next(),
            KeyCode::Left | KeyCode::Char('p') => self.prev_step(),
            KeyCode::Up | KeyCode::Char('k') => self.handle_up(),
            KeyCode::Down | KeyCode::Char('j') => self.handle_down(),
            KeyCode::Char(' ') => self.handle_space(),
            KeyCode::Char('r') => {
                // Retry health check
                if self.step == WizardStep::HealthCheck && !self.health_running {
                    self.trigger_health_check();
                }
            }
            _ => {}
        }
    }

    fn handle_input_key(&mut self, key: KeyCode) {
        // Handle data setup input mode
        if self.data_setup.input_mode {
            match key {
                KeyCode::Enter => {
                    match self.data_setup.sub_step {
                        DataSetupSubStep::EnterPath => {
                            self.data_setup.confirm_path();
                        }
                        DataSetupSubStep::EnterNamespace => {
                            self.data_setup.confirm_namespace();
                            // Start indexing
                            if self.data_setup.is_indexing() {
                                self.start_indexing_task();
                            }
                        }
                        _ => {}
                    }
                }
                KeyCode::Esc => {
                    self.data_setup.input_mode = false;
                    self.data_setup.input_buffer.clear();
                    self.data_setup.sub_step = DataSetupSubStep::SelectOption;
                }
                KeyCode::Backspace => {
                    self.data_setup.input_buffer.pop();
                }
                KeyCode::Char(c) => {
                    self.data_setup.input_buffer.push(c);
                }
                _ => {}
            }
            return;
        }

        // Handle settings or embedder input mode
        if self.input_mode {
            match key {
                KeyCode::Enter => {
                    if let Some(field) = self.editing_field {
                        // Handle embedder setup fields
                        if self.step == WizardStep::EmbedderSetup && self.embedder_state.use_manual
                        {
                            match field {
                                0 => self.embedder_state.manual_url = self.input_buffer.clone(),
                                1 => self.embedder_state.manual_model = self.input_buffer.clone(),
                                2 => {
                                    if let Ok(dim) = self.input_buffer.parse() {
                                        self.embedder_state.dimension = dim;
                                    }
                                }
                                _ => {}
                            }
                        } else {
                            self.set_field_value(field, self.input_buffer.clone());
                        }
                    }
                    self.input_mode = false;
                    self.editing_field = None;
                    self.input_buffer.clear();
                }
                KeyCode::Esc => {
                    // In manual embedder mode, go back to provider selection
                    if self.step == WizardStep::EmbedderSetup && self.embedder_state.use_manual {
                        self.embedder_state.use_manual = false;
                        self.focus = 0;
                    }
                    self.input_mode = false;
                    self.editing_field = None;
                    self.input_buffer.clear();
                }
                KeyCode::Backspace => {
                    self.input_buffer.pop();
                }
                KeyCode::Char(c) => {
                    self.input_buffer.push(c);
                }
                _ => {}
            }
        }
    }

    fn handle_enter(&mut self) {
        match self.step {
            WizardStep::EmbedderSetup => {
                self.handle_embedder_setup_enter();
            }
            WizardStep::MemexSettings => {
                // Enter edit mode for current field
                self.input_mode = true;
                self.editing_field = Some(self.focus);
                self.input_buffer = self.get_field_value(self.focus);
            }
            WizardStep::HostSelection => {
                if self.focus < self.hosts.len() {
                    self.toggle_host(self.focus);
                }
            }
            WizardStep::HealthCheck => {
                if !self.health_running {
                    self.trigger_health_check();
                }
            }
            WizardStep::DataSetup => {
                self.handle_data_setup_enter();
            }
            WizardStep::Summary => {
                // First write rmcp-memex config, then write host configs
                if !self.config_written
                    && let Err(e) = self.write_memex_config()
                {
                    self.messages.push(format!("[ERR] {}", e));
                }
                // Also write host configs
                if let Err(e) = self.write_configs() {
                    self.messages.push(format!("[ERR] {}", e));
                }
            }
            _ => {}
        }
    }

    fn handle_embedder_setup_enter(&mut self) {
        if self.embedder_state.use_manual {
            // In manual mode, edit the focused field
            self.input_mode = true;
            self.editing_field = Some(self.focus);
            self.input_buffer = match self.focus {
                0 => self.embedder_state.manual_url.clone(),
                1 => self.embedder_state.manual_model.clone(),
                2 => self.embedder_state.dimension.to_string(),
                _ => String::new(),
            };
        } else if self.focus < self.embedder_state.detected_providers.len() {
            // Select a detected provider
            let provider = self.embedder_state.detected_providers[self.focus].clone();
            self.embedder_state.dimension = provider.dimension();
            self.embedder_state.selected_provider = Some(provider);
        } else {
            // Switch to manual configuration (last option)
            self.embedder_state.use_manual = true;
            self.focus = 0;
        }
    }

    fn handle_data_setup_enter(&mut self) {
        match self.data_setup.sub_step {
            DataSetupSubStep::SelectOption => {
                self.data_setup.select_focused();
            }
            DataSetupSubStep::SelectImportMode => {
                let modes = ImportMode::all();
                if let Some(mode) = modes.get(self.data_setup.focus).cloned() {
                    self.data_setup.select_import_mode(mode);
                    // If import mode is selected, perform the import
                    if self.data_setup.is_complete()
                        && self.data_setup.option == DataSetupOption::ImportLanceDB
                    {
                        self.perform_import();
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_next(&mut self) {
        // For DataSetup, only proceed if complete or skip
        if self.step == WizardStep::DataSetup {
            if self.data_setup.is_complete() || self.data_setup.option == DataSetupOption::Skip {
                self.next_step();
            }
        } else if self.step == WizardStep::HealthCheck {
            // Allow proceeding even if health check failed (with warning)
            self.next_step();
        } else {
            self.next_step();
        }
    }

    fn handle_up(&mut self) {
        if self.focus > 0 {
            self.focus -= 1;
        }
        // Sync focus with data setup
        if self.step == WizardStep::DataSetup {
            self.data_setup.focus = self.focus;
        }
    }

    fn handle_down(&mut self) {
        let max = self.get_max_focus();
        if self.focus < max {
            self.focus += 1;
        }
        // Sync focus with data setup
        if self.step == WizardStep::DataSetup {
            self.data_setup.focus = self.focus;
        }
    }

    fn handle_space(&mut self) {
        if self.step == WizardStep::HostSelection && self.focus < self.hosts.len() {
            self.toggle_host(self.focus);
        }
    }

    fn get_max_focus(&self) -> usize {
        match self.step {
            WizardStep::EmbedderSetup => {
                if self.embedder_state.use_manual {
                    2 // URL, model, dimension
                } else {
                    // providers + manual option
                    self.embedder_state.detected_providers.len()
                }
            }
            WizardStep::MemexSettings => self.settings_field_count().saturating_sub(1),
            WizardStep::HostSelection => self.hosts.len().saturating_sub(1),
            WizardStep::DataSetup => match self.data_setup.sub_step {
                DataSetupSubStep::SelectOption => DataSetupOption::all().len().saturating_sub(1),
                DataSetupSubStep::SelectImportMode => ImportMode::all().len().saturating_sub(1),
                _ => 0,
            },
            _ => 0,
        }
    }

    /// Trigger the async health check
    pub fn trigger_health_check(&mut self) {
        self.health_running = true;
        self.health_status = Some("Running health checks...".to_string());
        self.messages.clear();

        // Also run the old basic check for binary version
        if let Ok(output) = std::process::Command::new(&self.binary_path)
            .arg("--version")
            .output()
            && output.status.success()
        {
            let version = String::from_utf8_lossy(&output.stdout);
            self.health_status = Some(format!("Binary: {} - Running checks...", version.trim()));
        }
    }

    /// Run the async health check (called from event loop)
    pub async fn run_async_health_check(&mut self) {
        // Quick connectivity check for selected provider
        if let Some(ref provider) = self.embedder_state.selected_provider {
            let url = format!("{}/v1/models", provider.base_url);
            if check_health(&url).await {
                self.messages
                    .push(format!("[OK] Provider {} is reachable", provider.base_url));
            } else {
                self.messages.push(format!(
                    "[WARN] Provider {} may be offline",
                    provider.base_url
                ));
            }
        }

        let checker = HealthChecker::new();
        let result = checker
            .run_all(&self.embedding_config, &self.memex_cfg.db_path)
            .await;

        self.health_result = Some(result.clone());
        self.health_running = false;

        // Update status based on results
        if result.all_passed() {
            self.health_status = Some("All health checks passed!".to_string());
        } else if result.any_failed() {
            self.health_status =
                Some("Some health checks failed. Review details below.".to_string());
        } else {
            self.health_status = Some("Health checks complete.".to_string());
        }
    }

    /// Start the indexing task
    fn start_indexing_task(&mut self) {
        if let Some(ref source_path) = self.data_setup.source_path
            && let Some(ref namespace) = self.data_setup.namespace
        {
            let path = PathBuf::from(shellexpand::tilde(source_path).to_string());
            let rx = start_indexing(
                path,
                namespace.clone(),
                self.embedding_config.clone(),
                self.memex_cfg.db_path.clone(),
            );
            self.index_progress_rx = Some(rx);
        }
    }

    /// Perform LanceDB import
    fn perform_import(&mut self) {
        if let Some(ref source_path) = self.data_setup.source_path {
            let source = PathBuf::from(shellexpand::tilde(source_path).to_string());
            let target = PathBuf::from(shellexpand::tilde(&self.memex_cfg.db_path).to_string());

            // Run import synchronously for now (it's mostly IO)
            let rt = tokio::runtime::Handle::try_current();
            if let Ok(handle) = rt {
                let mode = self.data_setup.import_mode.clone();
                let result = tokio::task::block_in_place(|| {
                    handle.block_on(import_lancedb(&source, &target, mode))
                });
                match result {
                    Ok(msg) => {
                        self.messages.push(format!("[OK] {}", msg));
                    }
                    Err(e) => {
                        self.messages.push(format!("[ERR] Import failed: {}", e));
                    }
                }
            } else {
                // Fallback for non-async context
                self.messages
                    .push("[INFO] Import will use config path directly".to_string());
            }
        }
    }

    /// Check for indexing progress updates
    pub fn poll_index_progress(&mut self) {
        if let Some(ref mut rx) = self.index_progress_rx {
            while let Ok(progress) = rx.try_recv() {
                self.data_setup.progress = Some(progress.clone());
                if progress.complete {
                    if let Some(ref error) = progress.error {
                        self.messages
                            .push(format!("[ERR] Indexing failed: {}", error));
                    } else {
                        self.messages.push(format!(
                            "[OK] Indexed {} files ({} skipped)",
                            progress.processed - progress.skipped,
                            progress.skipped
                        ));
                    }
                    self.data_setup.sub_step = DataSetupSubStep::Complete;
                    self.index_progress_rx = None;
                    break;
                }
            }
        }
    }

    /// Run provider detection asynchronously
    pub async fn run_provider_detection(&mut self) {
        if self.embedder_state.detecting {
            self.embedder_state.detected_providers = detect_providers().await;
            self.embedder_state.detecting = false;

            // Auto-select first usable provider
            if let Some(provider) = self
                .embedder_state
                .detected_providers
                .iter()
                .find(|p| p.is_usable())
            {
                self.embedder_state.selected_provider = Some(provider.clone());
                self.embedder_state.dimension = provider.dimension();
            }
        }
    }

    /// Generate the complete config TOML for rmcp-memex
    pub fn generate_config_toml(&self) -> String {
        let mut toml = String::new();

        // Header
        toml.push_str("# rmcp-memex configuration\n");
        toml.push_str(&format!(
            "# Generated by wizard on host: {}\n",
            self.memex_cfg.hostname
        ));
        toml.push_str(&format!(
            "# Path mode: {:?}\n\n",
            self.memex_cfg.db_path_mode
        ));

        // Database settings (use effective path which includes hostname suffix if per-host)
        toml.push_str("# Database configuration\n");
        toml.push_str(&format!(
            "db_path = \"{}\"\n",
            self.memex_cfg.effective_db_path()
        ));
        toml.push_str(&format!("cache_mb = {}\n", self.memex_cfg.cache_mb));
        toml.push_str(&format!("log_level = \"{}\"\n", self.memex_cfg.log_level));
        toml.push_str(&format!(
            "max_request_bytes = {}\n",
            self.memex_cfg.max_request_bytes
        ));

        // HTTP/SSE server configuration
        if let Some(port) = self.memex_cfg.http_port {
            toml.push_str("\n# HTTP/SSE server for multi-agent access\n");
            toml.push_str(&format!("http_port = {}\n", port));
        }
        toml.push('\n');

        // Embeddings configuration
        toml.push_str("# Embedding provider configuration\n");
        toml.push_str("[embeddings]\n");
        toml.push_str(&format!(
            "required_dimension = {}\n\n",
            self.embedder_state.dimension
        ));

        // Provider
        toml.push_str("[[embeddings.providers]]\n");
        if self.embedder_state.use_manual {
            toml.push_str("name = \"manual\"\n");
            toml.push_str(&format!(
                "base_url = \"{}\"\n",
                self.embedder_state.manual_url
            ));
            toml.push_str(&format!(
                "model = \"{}\"\n",
                self.embedder_state.manual_model
            ));
        } else if let Some(ref provider) = self.embedder_state.selected_provider {
            let name = match provider.kind {
                ProviderKind::Ollama => "ollama-local",
                ProviderKind::Mlx => "mlx-local",
                ProviderKind::OpenAICompat => "openai-compat",
                ProviderKind::Manual => "manual",
            };
            toml.push_str(&format!("name = \"{}\"\n", name));
            toml.push_str(&format!("base_url = \"{}\"\n", provider.base_url));
            toml.push_str(&format!(
                "model = \"{}\"\n",
                provider.model().unwrap_or("unknown")
            ));
        } else {
            // Fallback default
            toml.push_str("name = \"ollama-local\"\n");
            toml.push_str("base_url = \"http://localhost:11434\"\n");
            toml.push_str("model = \"qwen3-embedding:8b\"\n");
        }
        toml.push_str("priority = 1\n");
        toml.push_str("endpoint = \"/v1/embeddings\"\n");

        toml
    }

    /// Write rmcp-memex config file to disk
    pub fn write_memex_config(&mut self) -> Result<()> {
        if self.dry_run {
            self.messages
                .push("DRY RUN: Config would be written to:".to_string());
            self.messages
                .push("  ~/.rmcp-servers/rmcp-memex/config.toml".to_string());
            self.messages.push(String::new());
            self.messages.push("Generated config:".to_string());
            self.messages.push("---".to_string());
            for line in self.generate_config_toml().lines() {
                self.messages.push(format!("  {}", line));
            }
            self.messages.push("---".to_string());
            self.config_written = true;
            return Ok(());
        }

        // Create config directory
        let config_dir = shellexpand::tilde("~/.rmcp-servers/rmcp-memex").to_string();
        let config_path = format!("{}/config.toml", config_dir);

        std::fs::create_dir_all(&config_dir)?;

        // Backup existing config if present
        let config_file = PathBuf::from(&config_path);
        if config_file.exists() {
            let backup_path = format!("{}.bak.{}", config_path, timestamp());
            std::fs::copy(&config_file, &backup_path)?;
            self.messages
                .push(format!("[OK] Backup created: {}", backup_path));
        }

        // Write new config
        let toml_content = self.generate_config_toml();
        std::fs::write(&config_path, &toml_content)?;
        self.messages
            .push(format!("[OK] Config written: {}", config_path));

        // Create database directory if needed
        let db_path = shellexpand::tilde(&self.memex_cfg.db_path).to_string();
        if let Some(parent) = PathBuf::from(&db_path).parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
            self.messages
                .push(format!("[OK] Created directory: {}", parent.display()));
        }

        self.config_written = true;
        self.messages.push(String::new());
        self.messages.push("Configuration complete!".to_string());
        self.messages
            .push("Run 'rmcp_memex serve' to start the server.".to_string());

        Ok(())
    }
}

fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}

fn which_rmcp_memex() -> Option<String> {
    std::process::Command::new("which")
        .arg("rmcp_memex")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn init_terminal() -> Result<Tui> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn restore_terminal() -> Result<()> {
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Run the TUI wizard.
pub fn run_wizard(config: WizardConfig) -> Result<()> {
    let mut terminal = init_terminal()?;
    let mut app = App::new(config);

    let result = run_app(&mut terminal, &mut app);

    restore_terminal()?;
    result
}

fn run_app(terminal: &mut Tui, app: &mut App) -> Result<()> {
    use crate::tui::ui::render;

    // Get handle to existing runtime (from async main) or create new one
    let rt = match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle,
        Err(_) => {
            // No runtime exists, create one (shouldn't happen with async main)
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            // Leak to keep it alive - this is a fallback path
            Box::leak(Box::new(rt)).handle().clone()
        }
    };

    loop {
        terminal.draw(|f| render(f, app))?;

        // Poll for index progress updates
        app.poll_index_progress();

        // Handle async provider detection
        if app.embedder_state.detecting {
            let rt_clone = rt.clone();
            tokio::task::block_in_place(|| {
                rt_clone.block_on(async {
                    app.run_provider_detection().await;
                });
            });
        }

        // Handle async health check if triggered
        if app.health_running && app.health_result.is_none() {
            let rt_clone = rt.clone();
            tokio::task::block_in_place(|| {
                rt_clone.block_on(async {
                    app.run_async_health_check().await;
                });
            });
        }

        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            app.handle_key(key.code);
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
