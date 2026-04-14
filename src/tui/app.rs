//! TUI Wizard Application Logic
//!
//! Main application state and step management for the configuration wizard.
//! Implements the new wizard flow with EmbedderSetup as the first configuration step.

use crate::embeddings::{
    DEFAULT_REQUIRED_DIMENSION, EmbeddingConfig, ProviderConfig, probe_provider_dimension,
};
use crate::tui::detection::{
    DetectedProvider, ProviderKind, check_health, detect_providers, dimension_explanation,
};
use crate::tui::health::{HealthCheckResult, HealthChecker};
use crate::tui::host_detection::{
    DEFAULT_MUX_CONFIG_PATH, DEFAULT_MUX_SERVICE_NAME, DEFAULT_MUX_SOCKET_PATH, ExtendedHostKind,
    HostDetection, detect_extended_hosts, generate_extended_snippet, generate_extended_snippet_mux,
    write_extended_host_config, write_extended_host_config_mux, write_mux_service_config,
};
use crate::tui::indexer::{
    DataSetupOption, DataSetupState, DataSetupSubStep, FanOut, ImportMode, IndexControl,
    IndexEventSink, IndexTelemetrySnapshot, SharedIndexTelemetry, TracingSink, TuiTelemetrySink,
    collect_indexable_files, import_lancedb, new_index_telemetry, start_indexing, validate_path,
};
use crate::tui::monitor::{MonitorSnapshot, spawn_monitor};
use anyhow::{Result, anyhow};
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use reqwest::Client;
use std::io::{Stdout, stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

const DEFAULT_INDEX_PARALLELISM: usize = 4;
const DEFAULT_MEMEX_CONFIG_PATH: &str = "~/.rmcp-servers/rmcp-memex/config.toml";

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

/// How trustworthy the currently selected embedding dimension is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionTruth {
    /// No verified dimension yet — requires probe or manual entry.
    Pending,
    /// Verified by a live embedding probe against the provider.
    Probed,
    /// Explicitly set by the operator.
    Manual,
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
    /// Where the current dimension came from.
    pub dimension_truth: DimensionTruth,
    /// Whether a live dimension probe is currently pending.
    pub dimension_probe_in_flight: bool,
    /// Last live probe error, if any.
    pub dimension_probe_error: Option<String>,
    /// Provider config queued for the next live probe attempt.
    pub pending_dimension_probe: Option<ProviderConfig>,
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
            manual_model: String::new(),
            dimension: DEFAULT_REQUIRED_DIMENSION,
            dimension_truth: DimensionTruth::Pending,
            dimension_probe_in_flight: false,
            dimension_probe_error: None,
            pending_dimension_probe: None,
            use_manual: false,
        }
    }
}

impl EmbedderState {
    pub fn selected_model(&self) -> Option<String> {
        if self.use_manual {
            let model = self.manual_model.trim();
            if model.is_empty() {
                None
            } else {
                Some(model.to_string())
            }
        } else if let Some(ref detected) = self.selected_provider {
            detected
                .model()
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .map(ToOwned::to_owned)
        } else {
            None
        }
    }

    pub fn selected_base_url(&self) -> Option<&str> {
        if self.use_manual {
            let url = self.manual_url.trim();
            if url.is_empty() { None } else { Some(url) }
        } else {
            self.selected_provider
                .as_ref()
                .map(|provider| provider.base_url.trim())
                .filter(|url| !url.is_empty())
        }
    }

    pub fn dimension_display(&self) -> String {
        if self.dimension_probe_in_flight {
            return "probing...".to_string();
        }

        let suffix = match self.dimension_truth {
            DimensionTruth::Pending => "pending",
            DimensionTruth::Probed => "probed",
            DimensionTruth::Manual => "manual",
        };

        format!("{} [{}]", self.dimension, suffix)
    }

    /// Get dimension explanation text
    pub fn dimension_hint(&self) -> String {
        if self.dimension_probe_in_flight {
            return "Probing the provider for the actual vector size.".to_string();
        }

        if let Some(error) = &self.dimension_probe_error {
            let concise_error = error.lines().next().unwrap_or(error).trim();
            return match self.dimension_truth {
                DimensionTruth::Manual => format!(
                    "Manual override is active. Probe failed, but the operator-supplied dimension will be used. Probe error: {concise_error}"
                ),
                _ => format!(
                    "Live probe failed. Run Health Check or set the dimension manually before writing config. Probe error: {concise_error}"
                ),
            };
        }

        match self.dimension_truth {
            DimensionTruth::Pending => {
                if let Some(model) = self.selected_model() {
                    format!("No verified dimension for `{model}` yet. Run a probe or enter the dimension manually.")
                } else {
                    "Select an embedding model or enter one manually.".to_string()
                }
            }
            DimensionTruth::Probed => format!(
                "Verified live against the provider. {}",
                dimension_explanation(self.dimension, self.selected_model().as_deref())
            ),
            DimensionTruth::Manual => format!(
                "Set manually by the operator. {}",
                dimension_explanation(self.dimension, self.selected_model().as_deref())
            ),
        }
    }

    pub fn dimension_write_blocker(&self) -> Option<String> {
        if self.dimension_probe_in_flight {
            return Some(
                "Embedding dimension is still being probed. Wait for the live probe to finish or set a manual dimension.".to_string(),
            );
        }

        match self.dimension_truth {
            DimensionTruth::Pending => Some(
                "Embedding dimension has not been verified. Let the live probe succeed or enter the dimension manually before writing config.".to_string(),
            ),
            DimensionTruth::Probed | DimensionTruth::Manual => None,
        }
    }

    fn reset_probe_state(&mut self) {
        self.dimension_probe_in_flight = false;
        self.dimension_probe_error = None;
        self.pending_dimension_probe = None;
    }

    fn schedule_dimension_probe(&mut self, provider: ProviderConfig) {
        self.pending_dimension_probe = Some(provider);
        self.dimension_probe_in_flight = true;
        self.dimension_probe_error = None;
    }

    fn current_provider_config(&self) -> Option<ProviderConfig> {
        let model = self.selected_model()?;
        let base_url = self.selected_base_url()?.to_string();

        Some(ProviderConfig {
            name: if self.use_manual {
                "manual".to_string()
            } else if let Some(provider) = &self.selected_provider {
                match provider.kind {
                    ProviderKind::Ollama => "ollama-local".to_string(),
                    ProviderKind::Mlx => "mlx-local".to_string(),
                    ProviderKind::OpenAICompat => "openai-compat".to_string(),
                    ProviderKind::Manual => "manual".to_string(),
                }
            } else {
                "manual".to_string()
            },
            base_url,
            model,
            priority: 1,
            ..Default::default()
        })
    }

    fn refresh_manual_dimension_state(&mut self) {
        self.selected_provider = None;
        self.dimension_probe_error = None;
        self.dimension = DEFAULT_REQUIRED_DIMENSION;
        self.dimension_truth = DimensionTruth::Pending;

        let model = self.manual_model.trim();
        if model.is_empty() {
            self.pending_dimension_probe = None;
            self.dimension_probe_in_flight = false;
            return;
        }

        if let Some(provider) = self.current_provider_config() {
            self.schedule_dimension_probe(provider);
        } else {
            self.dimension_probe_in_flight = false;
            self.pending_dimension_probe = None;
        }
    }

    fn set_manual_dimension(&mut self, dimension: usize) {
        self.dimension = dimension;
        self.dimension_truth = DimensionTruth::Manual;
        self.reset_probe_state();
    }

    fn apply_detected_provider(&mut self, provider: DetectedProvider) {
        self.use_manual = false;
        self.selected_provider = Some(provider);
        self.dimension_probe_error = None;
        self.dimension = DEFAULT_REQUIRED_DIMENSION;
        self.dimension_truth = DimensionTruth::Pending;

        if let Some(provider) = self.current_provider_config() {
            self.schedule_dimension_probe(provider);
        } else {
            self.dimension_probe_in_flight = false;
            self.pending_dimension_probe = None;
        }
    }

    fn apply_probe_result(&mut self, result: Result<usize>) {
        self.dimension_probe_in_flight = false;

        match result {
            Ok(dimension) => {
                self.dimension = dimension;
                self.dimension_truth = DimensionTruth::Probed;
                self.dimension_probe_error = None;
            }
            Err(error) => {
                self.dimension_probe_error = Some(error.to_string());
            }
        }

        self.pending_dimension_probe = None;
    }

    /// Update embedding config from state
    pub fn build_embedding_config(&self) -> EmbeddingConfig {
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
                model: self.selected_model().unwrap_or_default(),
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

/// How hosts connect to rmcp-memex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeploymentMode {
    #[default]
    PerHostStdio,
    SharedMux,
}

/// Editable memex configuration.
#[derive(Debug, Clone)]
pub struct MemexCfg {
    pub db_path: String,
    pub cache_mb: usize,
    pub log_level: String,
    pub max_request_bytes: usize,
    /// Current machine hostname (auto-detected)
    pub hostname: String,
    /// Database path mode (shared vs per-host)
    pub db_path_mode: DbPathMode,
    /// HTTP/SSE server port (None = disabled, Some(port) = enabled)
    pub http_port: Option<u16>,
    /// Whether hosts launch rmcp-memex directly or via rmcp_mux_proxy.
    pub deployment_mode: DeploymentMode,
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
            hostname,
            db_path_mode: DbPathMode::Shared,
            http_port: None,
            deployment_mode: DeploymentMode::PerHostStdio,
        }
    }
}

impl MemexCfg {
    /// Get the effective database path (with hostname suffix if per-host mode)
    pub fn resolved_db_path(&self) -> String {
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
    pub config_path: String,
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
    /// Latest telemetry receiver for the indexing dashboard.
    pub telemetry_rx: Option<watch::Receiver<IndexTelemetrySnapshot>>,
    /// Latest system monitor receiver for the dashboard.
    pub monitor_rx: Option<watch::Receiver<MonitorSnapshot>>,
    /// Scheduler control sender.
    pub index_control_tx: Option<mpsc::Sender<IndexControl>>,
    /// Running indexing task.
    pub index_task: Option<JoinHandle<Result<()>>>,
    /// Running monitor sampler task.
    pub monitor_task: Option<JoinHandle<()>>,
    /// Background dimension probe task paired with its generation token.
    pub dimension_probe_task: Option<(u64, JoinHandle<Result<usize>>)>,
    /// Monotonic token that invalidates stale probe completions.
    pub dimension_probe_generation: u64,
    /// Current requested indexer parallelism.
    pub index_parallelism: usize,
    /// Whether the indexer is currently paused.
    pub index_paused: bool,
    /// Whether rmcp-memex config has been written
    pub config_written: bool,
    /// Resolved mux proxy command/path if available.
    pub mux_proxy_command: Option<String>,
}

fn which_mux_proxy() -> Option<String> {
    which_binary(&["rmcp_mux_proxy", "rmcp-mux-proxy"])
}

impl App {
    pub fn mux_proxy_on_path(&self) -> bool {
        self.mux_proxy_command.is_some()
    }

    pub fn mux_proxy_command(&self) -> Option<&str> {
        self.mux_proxy_command.as_deref()
    }

    fn required_mux_proxy_command(&self) -> Result<&str> {
        self.mux_proxy_command().ok_or_else(|| {
            anyhow!(
                "Shared mux mode requires `rmcp_mux_proxy` or `rmcp-mux-proxy` on PATH before writing host configs."
            )
        })
    }

    fn toggle_deployment_mode(&mut self) {
        self.memex_cfg.deployment_mode = match self.memex_cfg.deployment_mode {
            DeploymentMode::PerHostStdio => {
                if self.mux_proxy_on_path() {
                    DeploymentMode::SharedMux
                } else {
                    self.messages.push(
                        "[WARN] Shared mux mode is unavailable until `rmcp_mux_proxy` or `rmcp-mux-proxy` is on PATH.".to_string(),
                    );
                    DeploymentMode::PerHostStdio
                }
            }
            DeploymentMode::SharedMux => DeploymentMode::PerHostStdio,
        };
    }

    pub fn new(config: WizardConfig) -> Self {
        let WizardConfig {
            config_path,
            dry_run,
        } = config;
        let hosts = detect_extended_hosts();
        let binary_path = which_rmcp_memex().unwrap_or_else(|| "rmcp-memex".to_string());
        let embedder_state = EmbedderState::default();
        let embedding_config = embedder_state.build_embedding_config();
        let mux_proxy_command = which_mux_proxy();

        Self {
            step: WizardStep::Welcome,
            memex_cfg: MemexCfg::default(),
            config_path: config_path.unwrap_or_else(|| DEFAULT_MEMEX_CONFIG_PATH.to_string()),
            embedder_state,
            embedding_config,
            hosts,
            selected_hosts: Vec::new(),
            dry_run,
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
            telemetry_rx: None,
            monitor_rx: None,
            index_control_tx: None,
            index_task: None,
            monitor_task: None,
            dimension_probe_task: None,
            dimension_probe_generation: 0,
            index_parallelism: DEFAULT_INDEX_PARALLELISM,
            index_paused: false,
            config_written: false,
            mux_proxy_command,
        }
    }

    pub fn next_step(&mut self) {
        if let Some(next) = self.step.next() {
            // On leaving EmbedderSetup, update the embedding config
            if self.step == WizardStep::EmbedderSetup {
                self.refresh_embedding_config();
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
        let config_path = self.resolved_config_path();
        self.get_selected_hosts()
            .iter()
            .map(|(kind, _detection)| {
                let snippet = match self.memex_cfg.deployment_mode {
                    DeploymentMode::PerHostStdio => generate_extended_snippet(
                        *kind,
                        &self.binary_path,
                        &config_path,
                        self.memex_cfg.http_port,
                    ),
                    DeploymentMode::SharedMux => self
                        .mux_proxy_command()
                        .map(|proxy_command| {
                            generate_extended_snippet_mux(
                                *kind,
                                proxy_command,
                                DEFAULT_MUX_SOCKET_PATH,
                            )
                        })
                        .unwrap_or_else(|| {
                            "Shared mux unavailable: install `rmcp_mux_proxy` or `rmcp-mux-proxy` on PATH before generating host snippets.".to_string()
                        }),
                };
                (*kind, snippet)
            })
            .collect()
    }

    pub fn run_health_check(&mut self) {
        self.health_status = Some("Checking...".to_string());

        // Prefer the canonical binary name, but allow the legacy alias when present.
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
        self.messages
            .push(format!("[INFO] Config path: {}", self.config_path));

        // Check db_path (use effective path)
        let effective_path = self.memex_cfg.resolved_db_path();
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
        let config_path = self.resolved_config_path();
        let mux_proxy_command = if self.memex_cfg.deployment_mode == DeploymentMode::SharedMux {
            Some(self.required_mux_proxy_command()?.to_string())
        } else {
            None
        };

        if self.dry_run {
            self.messages.push("DRY RUN: No files written".to_string());
            self.messages.push(format!(
                "Host: {} | Path mode: {:?}",
                self.memex_cfg.hostname, self.memex_cfg.db_path_mode
            ));
            for &idx in &self.selected_hosts.clone() {
                if let Some((kind, detection)) = self.hosts.get(idx) {
                    let snippet = match self.memex_cfg.deployment_mode {
                        DeploymentMode::PerHostStdio => generate_extended_snippet(
                            *kind,
                            &self.binary_path,
                            &config_path,
                            self.memex_cfg.http_port,
                        ),
                        DeploymentMode::SharedMux => generate_extended_snippet_mux(
                            *kind,
                            mux_proxy_command
                                .as_deref()
                                .expect("mux proxy command must exist in shared mode"),
                            DEFAULT_MUX_SOCKET_PATH,
                        ),
                    };
                    self.messages.push(format!(
                        "Would write to {} ({}):\n{}",
                        kind.label(),
                        detection.path.display(),
                        snippet
                    ));
                }
            }
            if self.memex_cfg.deployment_mode == DeploymentMode::SharedMux {
                self.messages.push(format!(
                    "Would write mux service config to {}",
                    DEFAULT_MUX_CONFIG_PATH
                ));
            }
            return Ok(());
        }

        let mut success_count = 0;
        let mut error_count = 0;

        if self.memex_cfg.deployment_mode == DeploymentMode::SharedMux {
            match write_mux_service_config(
                &self.binary_path,
                &config_path,
                self.memex_cfg.http_port,
                self.memex_cfg.max_request_bytes,
                &self.memex_cfg.log_level,
            ) {
                Ok(result) => {
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
                Err(error) => {
                    self.messages
                        .push(format!("[ERR] rmcp-mux service config failed: {}", error));
                    return Err(error);
                }
            }
        }

        for &idx in &self.selected_hosts.clone() {
            if let Some((kind, _detection)) = self.hosts.get(idx) {
                let write_result = match self.memex_cfg.deployment_mode {
                    DeploymentMode::PerHostStdio => write_extended_host_config(
                        *kind,
                        &self.binary_path,
                        &config_path,
                        self.memex_cfg.http_port,
                    ),
                    DeploymentMode::SharedMux => write_extended_host_config_mux(
                        *kind,
                        mux_proxy_command
                            .as_deref()
                            .expect("mux proxy command must exist in shared mode"),
                        DEFAULT_MUX_SOCKET_PATH,
                    ),
                };

                match write_result {
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
                            .push(format!("[ERR] {} failed: {}", kind.label(), e));
                    }
                }
            }
        }

        if success_count > 0 {
            self.messages.push(format!(
                "\nConfiguration complete! {} host(s) configured.",
                success_count
            ));
            if self.memex_cfg.deployment_mode == DeploymentMode::SharedMux {
                self.messages.push(format!(
                    "Start the shared daemon with: rmcp_mux --config {} --service {}",
                    DEFAULT_MUX_CONFIG_PATH, DEFAULT_MUX_SERVICE_NAME
                ));
            }
        }
        if error_count > 0 {
            self.messages.push(format!(
                "Warning: {} host(s) failed to configure.",
                error_count
            ));
        }

        Ok(())
    }

    pub(crate) fn settings_field_count(&self) -> usize {
        7
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
            6 => match self.memex_cfg.deployment_mode {
                DeploymentMode::PerHostStdio => {
                    if self.mux_proxy_on_path() {
                        "Per-host (direct)".to_string()
                    } else {
                        "Per-host (shared unavailable)".to_string()
                    }
                }
                DeploymentMode::SharedMux => {
                    if self.mux_proxy_on_path() {
                        "Shared (mux)".to_string()
                    } else {
                        "Shared (blocked: proxy missing)".to_string()
                    }
                }
            },
            _ => String::new(),
        }
    }

    pub fn resolved_config_path(&self) -> String {
        let expanded = shellexpand::tilde(&self.config_path).to_string();
        let path = PathBuf::from(&expanded);
        if path.is_absolute() {
            expanded
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(path).display().to_string()
        } else {
            expanded
        }
    }

    fn refresh_embedding_config(&mut self) {
        self.embedding_config = self.embedder_state.build_embedding_config();
    }

    fn invalidate_dimension_probe_generation(&mut self) {
        self.dimension_probe_generation = self.dimension_probe_generation.wrapping_add(1);
    }

    fn cancel_dimension_probe_task(&mut self) {
        if let Some((_, handle)) = self.dimension_probe_task.take() {
            handle.abort();
        }
        self.invalidate_dimension_probe_generation();
    }

    fn start_dimension_probe_task(&mut self, rt: &tokio::runtime::Handle) {
        if self.dimension_probe_task.is_some() {
            return;
        }

        let Some(provider) = self.embedder_state.pending_dimension_probe.take() else {
            return;
        };

        let generation = self.dimension_probe_generation;
        let task = rt.spawn(async move {
            let client = Client::builder()
                .timeout(Duration::from_secs(8))
                .connect_timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_default();

            probe_provider_dimension(&client, &provider).await
        });

        self.dimension_probe_task = Some((generation, task));
    }

    fn apply_dimension_probe_completion(&mut self, generation: u64, result: Result<usize>) -> bool {
        if generation != self.dimension_probe_generation {
            return false;
        }

        self.embedder_state.apply_probe_result(result);
        self.refresh_embedding_config();
        true
    }

    fn poll_dimension_probe_task(&mut self, rt: &tokio::runtime::Handle) {
        let Some((generation, handle)) = self.dimension_probe_task.take() else {
            return;
        };

        if !handle.is_finished() {
            self.dimension_probe_task = Some((generation, handle));
            return;
        }

        let join_result = tokio::task::block_in_place(|| rt.block_on(handle));
        match join_result {
            Ok(result) => {
                let _ = self.apply_dimension_probe_completion(generation, result);
            }
            Err(error) if error.is_cancelled() => {}
            Err(error) => {
                let _ = self.apply_dimension_probe_completion(
                    generation,
                    Err(anyhow!("dimension probe task failed: {}", error)),
                );
            }
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
            6 => self.toggle_deployment_mode(),
            _ => {}
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) {
        // Handle input mode for settings and data setup
        if self.input_mode || self.data_setup.input_mode {
            self.handle_input_key(key);
            return;
        }

        if self.step == WizardStep::DataSetup
            && self.data_setup.sub_step == DataSetupSubStep::Indexing
        {
            match key {
                KeyCode::Char(' ') => {
                    let next = if self.index_paused {
                        IndexControl::Resume
                    } else {
                        IndexControl::Pause
                    };
                    if self.send_index_control(next) {
                        self.index_paused = !self.index_paused;
                    }
                    return;
                }
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    self.index_parallelism = self.index_parallelism.saturating_add(1);
                    if !self
                        .send_index_control(IndexControl::SetParallelism(self.index_parallelism))
                    {
                        self.index_parallelism = self.index_parallelism.saturating_sub(1).max(1);
                    }
                    return;
                }
                KeyCode::Char('-') => {
                    let previous = self.index_parallelism;
                    self.index_parallelism = self.index_parallelism.saturating_sub(1).max(1);
                    if previous != self.index_parallelism
                        && !self.send_index_control(IndexControl::SetParallelism(
                            self.index_parallelism,
                        ))
                    {
                        self.index_parallelism = previous;
                    }
                    return;
                }
                KeyCode::Char('s') => {
                    let _ = self.send_index_control(IndexControl::Stop);
                    return;
                }
                _ => {}
            }
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
                            self.cancel_dimension_probe_task();
                            match field {
                                0 => self.embedder_state.manual_url = self.input_buffer.clone(),
                                1 => {
                                    self.embedder_state.manual_model = self.input_buffer.clone();
                                    self.embedder_state.refresh_manual_dimension_state();
                                }
                                2 => {
                                    if let Ok(dim) = self.input_buffer.parse() {
                                        self.embedder_state.set_manual_dimension(dim);
                                    }
                                }
                                _ => {}
                            }
                            if field == 0 {
                                self.embedder_state.refresh_manual_dimension_state();
                            }
                            self.refresh_embedding_config();
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
            self.cancel_dimension_probe_task();
            let provider = self.embedder_state.detected_providers[self.focus].clone();
            self.embedder_state.apply_detected_provider(provider);
            self.refresh_embedding_config();
        } else {
            // Switch to manual configuration (last option)
            self.cancel_dimension_probe_task();
            self.embedder_state.use_manual = true;
            self.focus = 0;
            self.embedder_state.refresh_manual_dimension_state();
            self.refresh_embedding_config();
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
                    if self.data_setup.is_done()
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
            if self.data_setup.is_done() || self.data_setup.option == DataSetupOption::Skip {
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

    fn send_index_control(&mut self, control: IndexControl) -> bool {
        let Some(tx) = self.index_control_tx.clone() else {
            return false;
        };

        match tx.try_send(control) {
            Ok(()) => true,
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.messages
                    .push("[WARN] Index control queue is full; try again in a moment.".to_string());
                false
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                self.messages
                    .push("[WARN] Indexing controls are no longer available.".to_string());
                self.index_control_tx = None;
                false
            }
        }
    }

    pub fn current_index_telemetry(&self) -> Option<IndexTelemetrySnapshot> {
        self.telemetry_rx
            .as_ref()
            .map(|receiver| receiver.borrow().clone())
    }

    pub fn current_monitor_snapshot(&self) -> Option<MonitorSnapshot> {
        self.monitor_rx
            .as_ref()
            .map(|receiver| receiver.borrow().clone())
    }

    fn finish_indexing_from_snapshot(&mut self, snapshot: &IndexTelemetrySnapshot) {
        if let Some(error) = &snapshot.fatal_error {
            self.messages.push(format!(
                "[ERR] Indexing failed after {}/{} files: {}",
                snapshot.processed, snapshot.total, error
            ));
        } else if snapshot.stopped_early {
            self.messages.push(format!(
                "[WARN] Indexing stopped after {}/{} files ({} indexed, {} skipped, {} failed).",
                snapshot.processed,
                snapshot.total,
                snapshot.indexed,
                snapshot.skipped,
                snapshot.failed
            ));
        } else {
            self.messages.push(format!(
                "[OK] Indexing finished: {} indexed, {} skipped, {} failed, {} chunks.",
                snapshot.indexed, snapshot.skipped, snapshot.failed, snapshot.total_chunks
            ));
        }

        self.data_setup.sub_step = DataSetupSubStep::Complete;
        self.stop_indexing_tasks();
    }

    /// Trigger the async health check
    pub fn trigger_health_check(&mut self) {
        self.health_running = true;
        self.health_result = None;
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
        let effective_path = self.memex_cfg.resolved_db_path();
        let result = checker
            .run_all(&self.embedding_config, &effective_path)
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
        let Some(source_path) = self.data_setup.source_path.clone() else {
            return;
        };
        let Some(namespace) = self.data_setup.namespace.clone() else {
            return;
        };

        let path = match validate_path(&source_path) {
            Ok(path) => path,
            Err(error) => {
                self.data_setup.validation_error = Some(error.to_string());
                self.data_setup.sub_step = DataSetupSubStep::EnterPath;
                self.data_setup.input_mode = true;
                self.data_setup.input_buffer = source_path;
                return;
            }
        };

        let files = match collect_indexable_files(&path) {
            Ok(files) if !files.is_empty() => files,
            Ok(_) => {
                self.data_setup.validation_error =
                    Some("No indexable files found in the selected directory.".to_string());
                self.data_setup.sub_step = DataSetupSubStep::EnterPath;
                self.data_setup.input_mode = true;
                self.data_setup.input_buffer = source_path;
                return;
            }
            Err(error) => {
                self.data_setup.validation_error = Some(error.to_string());
                self.data_setup.sub_step = DataSetupSubStep::EnterPath;
                self.data_setup.input_mode = true;
                self.data_setup.input_buffer = source_path;
                return;
            }
        };

        self.data_setup.validation_error = None;
        self.messages.clear();
        self.stop_indexing_tasks();

        let total_files = files.len();
        let (telemetry_tx, telemetry_rx) = new_index_telemetry();
        let telemetry_tx: SharedIndexTelemetry = telemetry_tx;
        let tui_sink = Arc::new(TuiTelemetrySink::new(Arc::new(telemetry_tx)));
        let tracing_sink = Arc::new(TracingSink);
        let sinks: Vec<Arc<dyn IndexEventSink>> = vec![tui_sink, tracing_sink];
        let sink: Arc<dyn IndexEventSink> = Arc::new(FanOut::new(sinks));
        let (control_tx, control_rx) =
            mpsc::channel(crate::tui::indexer::INDEX_CONTROL_CHANNEL_CAPACITY);

        self.index_task = Some(start_indexing(
            path,
            files,
            namespace.clone(),
            self.embedding_config.clone(),
            self.memex_cfg.resolved_db_path(),
            sink,
            control_rx,
            self.index_parallelism,
        ));

        let (monitor_rx, monitor_task) = spawn_monitor(Duration::from_secs(1));

        self.telemetry_rx = Some(telemetry_rx);
        self.monitor_rx = Some(monitor_rx);
        self.index_control_tx = Some(control_tx);
        self.monitor_task = Some(monitor_task);
        self.index_paused = false;
        self.messages.push(format!(
            "[INFO] Indexing {} files into namespace {}.",
            total_files, namespace
        ));
    }

    fn stop_indexing_tasks(&mut self) {
        if let Some(handle) = self.index_task.take() {
            handle.abort();
        }
        if let Some(handle) = self.monitor_task.take() {
            handle.abort();
        }

        self.telemetry_rx = None;
        self.monitor_rx = None;
        self.index_control_tx = None;
        self.index_paused = false;
    }

    /// Perform LanceDB import
    fn perform_import(&mut self) {
        if let Some(ref source_path) = self.data_setup.source_path {
            let source = PathBuf::from(shellexpand::tilde(source_path).to_string());
            let target =
                PathBuf::from(shellexpand::tilde(&self.memex_cfg.resolved_db_path()).to_string());

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
                .cloned()
            {
                self.cancel_dimension_probe_task();
                self.embedder_state.apply_detected_provider(provider);
            } else {
                self.cancel_dimension_probe_task();
                self.embedder_state.reset_probe_state();
            }
            self.refresh_embedding_config();
        }
    }

    /// Generate the complete config TOML for rmcp-memex
    pub fn generate_config_toml(&self) -> String {
        const MODEL_PLACEHOLDER: &str = "<set-your-embedding-model>";
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
            self.memex_cfg.resolved_db_path()
        ));
        toml.push_str(&format!("cache_mb = {}\n", self.memex_cfg.cache_mb));
        toml.push_str(&format!("log_level = \"{}\"\n", self.memex_cfg.log_level));
        toml.push_str(&format!(
            "max_request_bytes = {}\n",
            self.memex_cfg.max_request_bytes
        ));

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
                self.embedder_state
                    .selected_model()
                    .unwrap_or_else(|| MODEL_PLACEHOLDER.to_string())
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
                provider.model().unwrap_or(MODEL_PLACEHOLDER)
            ));
        } else {
            // No provider selected yet: write an explicit placeholder instead of a false default.
            toml.push_str("name = \"ollama-local\"\n");
            toml.push_str("base_url = \"http://localhost:11434\"\n");
            toml.push_str(&format!("model = \"{}\"\n", MODEL_PLACEHOLDER));
        }
        toml.push_str("priority = 1\n");
        toml.push_str("endpoint = \"/v1/embeddings\"\n");

        toml
    }

    /// Write rmcp-memex config file to disk
    pub fn write_memex_config(&mut self) -> Result<()> {
        if self.embedder_state.selected_model().is_none() {
            return Err(anyhow!(
                "No embedding model selected. Pick a detected provider or enter a manual model before writing config."
            ));
        }

        if let Some(reason) = self.embedder_state.dimension_write_blocker() {
            return Err(anyhow!(reason));
        }

        if self.dry_run {
            self.messages
                .push("DRY RUN: Config would be written to:".to_string());
            self.messages.push(format!("  {}", self.config_path));
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

        let config_path = self.resolved_config_path();
        let config_file = PathBuf::from(&config_path);
        let config_dir = config_file.parent().ok_or_else(|| {
            anyhow!(
                "Cannot determine parent directory for config path {}",
                self.config_path
            )
        })?;
        std::fs::create_dir_all(config_dir)?;

        // Backup existing config if present
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
        let db_path = shellexpand::tilde(&self.memex_cfg.resolved_db_path()).to_string();
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
        if self.config_path == DEFAULT_MEMEX_CONFIG_PATH {
            self.messages
                .push("Run 'rmcp-memex serve' to start the server.".to_string());
        } else {
            self.messages.push(format!(
                "Run 'rmcp-memex serve --config {}' to start the server.",
                self.config_path
            ));
        }

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
    which_binary(&["rmcp-memex", "rmcp_memex"])
}

fn which_binary(candidates: &[&str]) -> Option<String> {
    candidates.iter().find_map(|binary| {
        std::process::Command::new("which")
            .arg(binary)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
    })
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
        app.start_dimension_probe_task(&rt);
        app.poll_dimension_probe_task(&rt);

        let current_telemetry = app.current_index_telemetry();
        if app.step == WizardStep::DataSetup
            && app.data_setup.sub_step == DataSetupSubStep::Indexing
            && let Some(snapshot) = current_telemetry.as_ref()
            && snapshot.complete
        {
            app.finish_indexing_from_snapshot(snapshot);
        }
        let current_monitor = app.current_monitor_snapshot();

        terminal.draw(|frame| {
            render(
                frame,
                app,
                current_telemetry.as_ref(),
                current_monitor.as_ref(),
            )
        })?;

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
            app.cancel_dimension_probe_task();
            app.stop_indexing_tasks();
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::detection::ProviderStatus;

    fn detected_provider(model: &str) -> DetectedProvider {
        DetectedProvider {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            port: 11434,
            models: vec![model.to_string()],
            suggested_model: Some(model.to_string()),
            status: ProviderStatus::Online(model.to_string()),
        }
    }

    #[test]
    fn detected_provider_selection_queues_probe_as_pending() {
        let mut state = EmbedderState::default();
        state.apply_detected_provider(detected_provider("qwen3-embedding:8b"));

        assert_eq!(state.dimension, DEFAULT_REQUIRED_DIMENSION);
        assert_eq!(state.dimension_truth, DimensionTruth::Pending);
        assert!(state.dimension_probe_in_flight);
        assert!(state.pending_dimension_probe.is_some());
        assert!(state.dimension_write_blocker().is_some());
    }

    #[test]
    fn manual_dimension_override_is_writable_without_probe() {
        let mut state = EmbedderState {
            use_manual: true,
            manual_url: "http://localhost:11434".to_string(),
            manual_model: "custom-embed".to_string(),
            ..EmbedderState::default()
        };

        state.set_manual_dimension(1536);

        assert_eq!(state.dimension, 1536);
        assert_eq!(state.dimension_truth, DimensionTruth::Manual);
        assert!(!state.dimension_probe_in_flight);
        assert!(state.dimension_write_blocker().is_none());
    }

    #[test]
    fn unknown_manual_model_without_probe_stays_blocked() {
        let mut state = EmbedderState {
            use_manual: true,
            manual_model: "custom-embed".to_string(),
            manual_url: String::new(),
            ..EmbedderState::default()
        };

        state.refresh_manual_dimension_state();

        assert_eq!(state.dimension_truth, DimensionTruth::Pending);
        assert!(state.dimension_write_blocker().is_some());
    }

    #[test]
    fn stale_probe_completion_cannot_override_newer_manual_choice() {
        let mut app = App::new(WizardConfig::default());
        app.embedder_state
            .apply_detected_provider(detected_provider("qwen3-embedding:8b"));
        app.refresh_embedding_config();

        app.dimension_probe_generation = 5;
        app.cancel_dimension_probe_task();
        app.embedder_state.set_manual_dimension(1536);
        app.refresh_embedding_config();

        let applied = app.apply_dimension_probe_completion(5, Ok(4096));

        assert!(!applied);
        assert_eq!(app.embedder_state.dimension, 1536);
        assert_eq!(app.embedder_state.dimension_truth, DimensionTruth::Manual);
        assert_eq!(app.embedding_config.required_dimension, 1536);
    }

    #[test]
    fn shared_mux_write_requires_resolved_proxy_command() {
        let mut app = App::new(WizardConfig {
            dry_run: true,
            ..WizardConfig::default()
        });
        app.memex_cfg.deployment_mode = DeploymentMode::SharedMux;
        app.mux_proxy_command = None;

        let error = app
            .write_configs()
            .expect_err("shared mux should be blocked");
        assert!(
            error
                .to_string()
                .contains("Shared mux mode requires `rmcp_mux_proxy` or `rmcp-mux-proxy` on PATH")
        );
    }

    #[test]
    fn deployment_mode_toggle_without_proxy_stays_direct_and_warns() {
        let mut app = App::new(WizardConfig::default());
        app.mux_proxy_command = None;

        app.set_field_value(6, String::new());

        assert_eq!(app.memex_cfg.deployment_mode, DeploymentMode::PerHostStdio);
        assert_eq!(
            app.get_field_value(6),
            "Per-host (shared unavailable)".to_string()
        );
        assert!(
            app.messages
                .last()
                .expect("warning message")
                .contains("Shared mux mode is unavailable")
        );
    }

    #[test]
    fn deployment_mode_toggle_with_proxy_enables_shared_mux() {
        let mut app = App::new(WizardConfig::default());
        app.mux_proxy_command = Some("/usr/local/bin/rmcp-mux-proxy".to_string());

        app.set_field_value(6, String::new());

        assert_eq!(app.memex_cfg.deployment_mode, DeploymentMode::SharedMux);
        assert_eq!(app.get_field_value(6), "Shared (mux)".to_string());
    }
}
