//! Host detection module for MCP server configurations.
//!
//! Scans known locations for MCP host configurations (Codex, Cursor, Claude Desktop, JetBrains).
//! Also provides config writing functionality for the wizard.

use crate::common::{HostFormat, HostKind};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// MCP SERVER ENTRIES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerEntry {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

pub const DEFAULT_MUX_SERVICE_NAME: &str = "rmcp-memex";
pub const DEFAULT_MUX_SOCKET_PATH: &str = "~/.rmcp-servers/rmcp-memex/sockets/main.sock";
pub const DEFAULT_MUX_CONFIG_PATH: &str = "~/.rmcp-servers/rmcp-memex/mux_config.toml";
const DEFAULT_MUX_STATUS_PATH: &str = "~/.rmcp-servers/rmcp-memex/status/main.json";
const RMCP_MEMEX_SERVER_NAME: &str = "rmcp_memex";
const MUX_MAX_ACTIVE_CLIENTS: usize = 5;
const MUX_REQUEST_TIMEOUT_MS: u64 = 30_000;
const MUX_RESTART_BACKOFF_MS: u64 = 1_000;
const MUX_RESTART_BACKOFF_MAX_MS: u64 = 30_000;
const MUX_MAX_RESTARTS: u64 = 5;

#[derive(Debug, Clone, Serialize)]
struct MuxConfigFile {
    servers: BTreeMap<String, MuxServiceConfig>,
}

#[derive(Debug, Clone, Serialize)]
struct MuxServiceConfig {
    socket: String,
    cmd: String,
    args: Vec<String>,
    max_active_clients: usize,
    max_request_bytes: usize,
    request_timeout_ms: u64,
    restart_backoff_ms: u64,
    restart_backoff_max_ms: u64,
    max_restarts: u64,
    lazy_start: bool,
    tray: bool,
    service_name: String,
    log_level: String,
    status_file: String,
}

#[derive(Debug, Clone)]
pub struct HostDetection {
    pub kind: HostKind,
    pub path: PathBuf,
    pub format: HostFormat,
    pub exists: bool,
    pub has_rmcp_memex: bool,
    pub servers: Vec<McpServerEntry>,
}

impl HostDetection {
    pub fn status_icon(&self) -> &'static str {
        if !self.exists {
            "[ ]"
        } else if self.has_rmcp_memex {
            "[x]"
        } else {
            "[~]"
        }
    }

    pub fn status_text(&self) -> &'static str {
        if !self.exists {
            "Not found"
        } else if self.has_rmcp_memex {
            "Configured"
        } else {
            "Detected (no memex server entry)"
        }
    }
}

fn matches_memex_server(entry: &McpServerEntry) -> bool {
    entry.name.contains("rmcp_memex")
        || entry.name.contains("rmcp-memex")
        || entry.command.contains("rmcp_memex")
        || entry.command.contains("rmcp-memex")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

fn expand_home_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(stripped);
    }

    PathBuf::from(path)
}

/// Extended host kind that includes hosts not in rmcp-common
/// (ClaudeCode and Junie are specific to rmcp-memex wizard)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendedHostKind {
    /// Standard hosts from rmcp-common
    Standard(HostKind),
    /// Claude Code CLI (~/.claude.json)
    ClaudeCode,
    /// Junie AI (~/.junie/mcp.json)
    Junie,
}

impl ExtendedHostKind {
    pub fn label(&self) -> &'static str {
        match self {
            ExtendedHostKind::Standard(k) => k.display_name(),
            ExtendedHostKind::ClaudeCode => "Claude Code",
            ExtendedHostKind::Junie => "Junie",
        }
    }
}

fn get_host_config_path(kind: HostKind) -> Option<(PathBuf, HostFormat)> {
    let home = home_dir()?;

    match kind {
        HostKind::Codex => Some((home.join(".codex/config.toml"), HostFormat::Toml)),
        HostKind::Cursor => {
            #[cfg(target_os = "macos")]
            let path = home.join(
                "Library/Application Support/Cursor/User/globalStorage/cursor.mcp/config.json",
            );
            #[cfg(target_os = "linux")]
            let path = home.join(".config/Cursor/User/globalStorage/cursor.mcp/config.json");
            #[cfg(target_os = "windows")]
            let path =
                home.join("AppData/Roaming/Cursor/User/globalStorage/cursor.mcp/config.json");
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            let path = home.join(".config/Cursor/config.json");
            Some((path, HostFormat::Json))
        }
        HostKind::Claude => {
            #[cfg(target_os = "macos")]
            let path = home.join("Library/Application Support/Claude/claude_desktop_config.json");
            #[cfg(target_os = "linux")]
            let path = home.join(".config/Claude/claude_desktop_config.json");
            #[cfg(target_os = "windows")]
            let path = home.join("AppData/Roaming/Claude/claude_desktop_config.json");
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            let path = home.join(".config/Claude/claude_desktop_config.json");
            Some((path, HostFormat::Json))
        }
        HostKind::JetBrains => {
            // JetBrains uses a common MCP config location
            #[cfg(target_os = "macos")]
            let path = home.join("Library/Application Support/JetBrains/mcp.json");
            #[cfg(target_os = "linux")]
            let path = home.join(".config/JetBrains/mcp.json");
            #[cfg(target_os = "windows")]
            let path = home.join("AppData/Roaming/JetBrains/mcp.json");
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            let path = home.join(".config/JetBrains/mcp.json");
            Some((path, HostFormat::Json))
        }
        HostKind::VSCode => {
            #[cfg(target_os = "macos")]
            let path = home.join("Library/Application Support/Code/User/globalStorage/anthropic.claude-vscode/settings/cline_mcp_settings.json");
            #[cfg(target_os = "linux")]
            let path = home.join(".config/Code/User/globalStorage/anthropic.claude-vscode/settings/cline_mcp_settings.json");
            #[cfg(target_os = "windows")]
            let path = home.join("AppData/Roaming/Code/User/globalStorage/anthropic.claude-vscode/settings/cline_mcp_settings.json");
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            let path = home.join(".config/Code/cline_mcp_settings.json");
            Some((path, HostFormat::Json))
        }
        HostKind::Unknown => None,
    }
}

/// Get config path for extended host kinds (including ClaudeCode and Junie)
pub fn get_extended_host_config_path(kind: ExtendedHostKind) -> Option<(PathBuf, HostFormat)> {
    let home = home_dir()?;

    match kind {
        ExtendedHostKind::Standard(k) => get_host_config_path(k),
        ExtendedHostKind::ClaudeCode => Some((home.join(".claude.json"), HostFormat::Json)),
        ExtendedHostKind::Junie => Some((home.join(".junie/mcp.json"), HostFormat::Json)),
    }
}

fn parse_toml_mcp_servers(content: &str) -> Vec<McpServerEntry> {
    let mut servers = Vec::new();

    if let Ok(value) = content.parse::<toml::Value>()
        && let Some(mcp_servers) = value.get("mcp_servers").and_then(|v| v.as_table())
    {
        for (name, config) in mcp_servers {
            let command = config
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let args = config
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let env = config
                .get("env")
                .and_then(|v| v.as_table())
                .map(|t| {
                    t.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();

            servers.push(McpServerEntry {
                name: name.clone(),
                command,
                args,
                env,
            });
        }
    }

    servers
}

fn parse_json_mcp_servers(content: &str) -> Vec<McpServerEntry> {
    let mut servers = Vec::new();

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
        let mcp_servers = value.get("mcpServers").or_else(|| value.get("mcp_servers"));

        if let Some(mcp_obj) = mcp_servers.and_then(|v| v.as_object()) {
            for (name, config) in mcp_obj {
                let command = config
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let args = config
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let env = config
                    .get("env")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();

                servers.push(McpServerEntry {
                    name: name.clone(),
                    command,
                    args,
                    env,
                });
            }
        }
    }

    servers
}

fn detect_single_host(kind: HostKind) -> Option<HostDetection> {
    let (path, format) = get_host_config_path(kind)?;
    let exists = path.exists();

    let (has_rmcp_memex, servers) = if exists {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let servers = match format {
                HostFormat::Toml => parse_toml_mcp_servers(&content),
                HostFormat::Json => parse_json_mcp_servers(&content),
            };
            let has_rmcp = servers.iter().any(matches_memex_server);
            (has_rmcp, servers)
        } else {
            (false, Vec::new())
        }
    } else {
        (false, Vec::new())
    };

    Some(HostDetection {
        kind,
        path,
        format,
        exists,
        has_rmcp_memex,
        servers,
    })
}

/// Detect all known MCP host configurations.
pub fn detect_hosts() -> Vec<HostDetection> {
    let kinds = [
        HostKind::Codex,
        HostKind::Cursor,
        HostKind::Claude,
        HostKind::JetBrains,
        HostKind::VSCode,
    ];

    kinds
        .iter()
        .filter_map(|&k| detect_single_host(k))
        .collect()
}

fn direct_command_args(config_path: &str, http_port: Option<u16>) -> Vec<String> {
    let mut args = vec!["serve".to_string()];
    if let Some(port) = http_port {
        args.push("--http-port".to_string());
        args.push(port.to_string());
    }
    args.push("--config".to_string());
    args.push(config_path.to_string());
    args
}

fn proxy_command_args(sock_path: &str) -> Vec<String> {
    vec!["--socket".to_string(), sock_path.to_string()]
}

fn build_server_entry(command: &str, args: Vec<String>) -> McpServerEntry {
    McpServerEntry {
        name: RMCP_MEMEX_SERVER_NAME.to_string(),
        command: command.to_string(),
        args,
        env: HashMap::new(),
    }
}

fn build_direct_host_entry(
    binary_path: &str,
    config_path: &str,
    http_port: Option<u16>,
) -> McpServerEntry {
    build_server_entry(binary_path, direct_command_args(config_path, http_port))
}

fn build_mux_host_entry(proxy_command: &str, sock_path: &str) -> McpServerEntry {
    build_server_entry(proxy_command, proxy_command_args(sock_path))
}

fn entry_description(entry: &McpServerEntry) -> &'static str {
    if entry.command.contains("rmcp_mux_proxy") || entry.command.contains("rmcp-mux-proxy") {
        "RAG memory via shared rmcp-mux proxy"
    } else {
        "RAG memory with vector search"
    }
}

fn json_server_config(entry: &McpServerEntry) -> serde_json::Value {
    let mut server = serde_json::Map::new();
    server.insert(
        "command".to_string(),
        serde_json::Value::String(entry.command.clone()),
    );
    server.insert(
        "args".to_string(),
        serde_json::Value::Array(
            entry
                .args
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    if !entry.env.is_empty() {
        server.insert(
            "env".to_string(),
            serde_json::Value::Object(
                entry
                    .env
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect(),
            ),
        );
    }
    server.insert(
        "description".to_string(),
        serde_json::Value::String(entry_description(entry).to_string()),
    );

    serde_json::Value::Object(server)
}

fn toml_server_config(entry: &McpServerEntry) -> toml::Value {
    let mut server = toml::map::Map::new();
    server.insert(
        "command".to_string(),
        toml::Value::String(entry.command.clone()),
    );
    server.insert(
        "args".to_string(),
        toml::Value::Array(
            entry
                .args
                .iter()
                .cloned()
                .map(toml::Value::String)
                .collect(),
        ),
    );
    if !entry.env.is_empty() {
        let env = entry
            .env
            .iter()
            .map(|(k, v)| (k.clone(), toml::Value::String(v.clone())))
            .collect();
        server.insert("env".to_string(), toml::Value::Table(env));
    }

    toml::Value::Table(server)
}

fn render_snippet(format: HostFormat, entry: &McpServerEntry) -> Result<String> {
    match format {
        HostFormat::Json => {
            let mut servers = serde_json::Map::new();
            servers.insert(entry.name.clone(), json_server_config(entry));
            let mut root = serde_json::Map::new();
            root.insert("mcpServers".to_string(), serde_json::Value::Object(servers));
            serde_json::to_string_pretty(&serde_json::Value::Object(root))
                .with_context(|| "Failed to serialize JSON snippet")
        }
        HostFormat::Toml => {
            let mut servers = toml::map::Map::new();
            servers.insert(entry.name.clone(), toml_server_config(entry));
            let mut root = toml::map::Map::new();
            root.insert("mcp_servers".to_string(), toml::Value::Table(servers));
            toml::to_string_pretty(&toml::Value::Table(root))
                .with_context(|| "Failed to serialize TOML snippet")
        }
    }
}

/// Generate a config snippet for an extended host kind.
pub fn generate_extended_snippet(
    kind: ExtendedHostKind,
    binary_path: &str,
    config_path: &str,
    http_port: Option<u16>,
) -> String {
    let Some((_, format)) = get_extended_host_config_path(kind) else {
        return String::new();
    };

    render_snippet(
        format,
        &build_direct_host_entry(binary_path, config_path, http_port),
    )
    .unwrap_or_default()
}

/// Result of writing a host config
#[derive(Debug)]
pub struct WriteResult {
    pub host_name: String,
    pub config_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub created: bool,
}

/// Generate a backup timestamp
fn backup_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}

/// Create a backup of an existing config file
fn create_backup(path: &Path) -> Result<PathBuf> {
    use crate::path_utils::validate_read_path;

    // Validate source path is safe to read
    let safe_src = validate_read_path(path).with_context(|| {
        format!(
            "Cannot backup: source path validation failed for {}",
            path.display()
        )
    })?;

    let backup_path = PathBuf::from(format!("{}.bak.{}", safe_src.display(), backup_timestamp()));

    // Atomic validated copy: validates both paths and copies in one step
    let safe_dst = crate::path_utils::safe_copy(&safe_src, &backup_path)
        .with_context(|| format!("Failed to create backup of {}", safe_src.display()))?;
    Ok(safe_dst)
}

/// Merge the rmcp_memex host entry into existing JSON config.
fn merge_json_config(existing_content: &str, entry: &McpServerEntry) -> Result<String> {
    let mut config: serde_json::Value = if existing_content.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(existing_content)
            .with_context(|| "Failed to parse existing JSON config")?
    };

    // Ensure mcpServers object exists
    if config.get("mcpServers").is_none() {
        config["mcpServers"] = serde_json::json!({});
    }

    // Add or update rmcp_memex entry
    config["mcpServers"][entry.name.as_str()] = json_server_config(entry);

    serde_json::to_string_pretty(&config).with_context(|| "Failed to serialize JSON config")
}

/// Merge the rmcp_memex host entry into existing TOML config.
fn merge_toml_config(existing_content: &str, entry: &McpServerEntry) -> Result<String> {
    let mut config: toml::Value = if existing_content.trim().is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        existing_content
            .parse()
            .with_context(|| "Failed to parse existing TOML config")?
    };

    // Ensure mcp_servers table exists
    let table = config.as_table_mut().expect("root must be a table");
    if !table.contains_key("mcp_servers") {
        table.insert(
            "mcp_servers".to_string(),
            toml::Value::Table(toml::map::Map::new()),
        );
    }

    // Add or update rmcp_memex entry
    if let Some(mcp_servers) = table.get_mut("mcp_servers").and_then(|v| v.as_table_mut()) {
        mcp_servers.insert(entry.name.clone(), toml_server_config(entry));
    }

    Ok(toml::to_string_pretty(&config)?)
}

fn write_host_config_entry(
    host_name: String,
    path: &Path,
    format: HostFormat,
    exists: bool,
    entry: &McpServerEntry,
) -> Result<WriteResult> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    // Create backup if file exists
    let backup_path = if exists {
        Some(create_backup(path)?)
    } else {
        None
    };

    use crate::path_utils::validate_write_path;

    // Read existing content or use empty string
    let existing_content = if exists {
        let (_safe_path, content) = crate::path_utils::safe_read_to_string(&path.to_string_lossy())
            .with_context(|| format!("Cannot read config: {}", path.display()))?;
        content
    } else {
        String::new()
    };

    // Merge config based on format
    let new_content = match format {
        HostFormat::Json => merge_json_config(&existing_content, entry)?,
        HostFormat::Toml => merge_toml_config(&existing_content, entry)?,
    };

    // Validate path before writing
    let safe_write_path = validate_write_path(path).with_context(|| {
        format!(
            "Cannot write config: path validation failed for {}",
            path.display()
        )
    })?;

    std::fs::write(&safe_write_path, &new_content)
        .with_context(|| format!("Failed to write config to {}", safe_write_path.display()))?;

    Ok(WriteResult {
        host_name,
        config_path: path.to_path_buf(),
        backup_path,
        created: !exists,
    })
}

/// Write config for an extended host kind (including ClaudeCode and Junie)
pub fn write_extended_host_config(
    kind: ExtendedHostKind,
    binary_path: &str,
    config_path: &str,
    http_port: Option<u16>,
) -> Result<WriteResult> {
    let (path, format) =
        get_extended_host_config_path(kind).ok_or_else(|| anyhow::anyhow!("Unknown host kind"))?;
    let entry = build_direct_host_entry(binary_path, config_path, http_port);
    write_host_config_entry(
        kind.label().to_string(),
        &path,
        format,
        path.exists(),
        &entry,
    )
}

/// Detect all extended hosts (including ClaudeCode and Junie)
pub fn detect_extended_hosts() -> Vec<(ExtendedHostKind, HostDetection)> {
    let mut results = Vec::new();

    // Standard hosts
    for kind in [
        HostKind::Codex,
        HostKind::Cursor,
        HostKind::Claude,
        HostKind::JetBrains,
        HostKind::VSCode,
    ] {
        if let Some(detection) = detect_single_host(kind) {
            results.push((ExtendedHostKind::Standard(kind), detection));
        }
    }

    // Extended hosts (ClaudeCode, Junie)
    for ext_kind in [ExtendedHostKind::ClaudeCode, ExtendedHostKind::Junie] {
        if let Some((path, format)) = get_extended_host_config_path(ext_kind) {
            let exists = path.exists();
            let (has_rmcp_memex, servers) = if exists {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let servers = parse_json_mcp_servers(&content);
                    let has_rmcp = servers.iter().any(matches_memex_server);
                    (has_rmcp, servers)
                } else {
                    (false, Vec::new())
                }
            } else {
                (false, Vec::new())
            };

            results.push((
                ext_kind,
                HostDetection {
                    kind: HostKind::Unknown,
                    path,
                    format,
                    exists,
                    has_rmcp_memex,
                    servers,
                },
            ));
        }
    }

    results
}

pub fn generate_extended_snippet_mux(
    kind: ExtendedHostKind,
    proxy_command: &str,
    sock_path: &str,
) -> String {
    let Some((_, format)) = get_extended_host_config_path(kind) else {
        return String::new();
    };

    render_snippet(format, &build_mux_host_entry(proxy_command, sock_path)).unwrap_or_default()
}

pub fn write_extended_host_config_mux(
    kind: ExtendedHostKind,
    proxy_command: &str,
    sock_path: &str,
) -> Result<WriteResult> {
    let (path, format) =
        get_extended_host_config_path(kind).ok_or_else(|| anyhow::anyhow!("Unknown host kind"))?;
    write_host_config_entry(
        kind.label().to_string(),
        &path,
        format,
        path.exists(),
        &build_mux_host_entry(proxy_command, sock_path),
    )
}

fn build_mux_service_config_toml(
    binary_path: &str,
    config_path: &str,
    http_port: Option<u16>,
    max_request_bytes: usize,
    log_level: &str,
) -> Result<String> {
    let mut servers = BTreeMap::new();
    servers.insert(
        DEFAULT_MUX_SERVICE_NAME.to_string(),
        MuxServiceConfig {
            socket: DEFAULT_MUX_SOCKET_PATH.to_string(),
            cmd: binary_path.to_string(),
            args: direct_command_args(config_path, http_port),
            max_active_clients: MUX_MAX_ACTIVE_CLIENTS,
            max_request_bytes,
            request_timeout_ms: MUX_REQUEST_TIMEOUT_MS,
            restart_backoff_ms: MUX_RESTART_BACKOFF_MS,
            restart_backoff_max_ms: MUX_RESTART_BACKOFF_MAX_MS,
            max_restarts: MUX_MAX_RESTARTS,
            lazy_start: false,
            tray: false,
            service_name: DEFAULT_MUX_SERVICE_NAME.to_string(),
            log_level: log_level.to_string(),
            status_file: DEFAULT_MUX_STATUS_PATH.to_string(),
        },
    );

    toml::to_string_pretty(&MuxConfigFile { servers })
        .with_context(|| "Failed to serialize mux service config")
}

pub fn write_mux_service_config(
    binary_path: &str,
    config_path: &str,
    http_port: Option<u16>,
    max_request_bytes: usize,
    log_level: &str,
) -> Result<WriteResult> {
    let config_file = expand_home_path(DEFAULT_MUX_CONFIG_PATH);
    let socket_dir = expand_home_path(DEFAULT_MUX_SOCKET_PATH)
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("Invalid mux socket path"))?;
    let status_dir = expand_home_path(DEFAULT_MUX_STATUS_PATH)
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("Invalid mux status path"))?;

    if let Some(parent) = config_file.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }
    if !socket_dir.exists() {
        std::fs::create_dir_all(&socket_dir)
            .with_context(|| format!("Failed to create directory {}", socket_dir.display()))?;
    }
    if !status_dir.exists() {
        std::fs::create_dir_all(&status_dir)
            .with_context(|| format!("Failed to create directory {}", status_dir.display()))?;
    }

    let exists = config_file.exists();
    let backup_path = if exists {
        Some(create_backup(&config_file)?)
    } else {
        None
    };

    use crate::path_utils::validate_write_path;

    let content = build_mux_service_config_toml(
        binary_path,
        config_path,
        http_port,
        max_request_bytes,
        log_level,
    )?;
    let safe_write_path = validate_write_path(&config_file).with_context(|| {
        format!(
            "Cannot write mux service config: path validation failed for {}",
            config_file.display()
        )
    })?;
    std::fs::write(&safe_write_path, content).with_context(|| {
        format!(
            "Failed to write mux service config to {}",
            safe_write_path.display()
        )
    })?;

    Ok(WriteResult {
        host_name: "rmcp-mux service".to_string(),
        config_path: config_file,
        backup_path,
        created: !exists,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_toml_mcp_servers() {
        let toml_content = r#"
[mcp_servers.rmcp_memex]
command = "/usr/local/bin/rmcp_memex"
args = ["--db-path", "~/.rmcp/db"]

[mcp_servers.other_server]
command = "other"
"#;
        let servers = parse_toml_mcp_servers(toml_content);
        assert_eq!(servers.len(), 2);
        assert!(servers.iter().any(|s| s.name == "rmcp_memex"));
    }

    #[test]
    fn test_parse_json_mcp_servers() {
        let json_content = r#"{
  "mcpServers": {
    "rmcp_memex": {
      "command": "/usr/local/bin/rmcp_memex",
      "args": ["--db-path", "~/.rmcp/db"]
    }
  }
}"#;
        let servers = parse_json_mcp_servers(json_content);
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "rmcp_memex");
    }

    #[test]
    fn test_matches_memex_server_accepts_canonical_binary_name() {
        let entry = McpServerEntry {
            name: "custom".to_string(),
            command: "/usr/local/bin/rmcp-memex".to_string(),
            args: vec!["serve".to_string()],
            env: HashMap::new(),
        };

        assert!(matches_memex_server(&entry));
    }

    #[test]
    fn test_generate_toml_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Standard(HostKind::Codex),
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            None,
        );
        assert!(snippet.contains("[mcp_servers.rmcp_memex]"));
        assert!(snippet.contains("/usr/bin/rmcp-memex"));
        assert!(snippet.contains("--config"));
    }

    #[test]
    fn test_generate_json_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Standard(HostKind::Claude),
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            None,
        );
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
        assert!(snippet.contains("/usr/bin/rmcp-memex"));
        assert!(snippet.contains("--config"));
    }

    #[test]
    fn test_generate_extended_claude_code_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::ClaudeCode,
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            None,
        );
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
        assert!(snippet.contains("/usr/bin/rmcp-memex"));
        assert!(snippet.contains("--config"));
    }

    #[test]
    fn test_generate_extended_junie_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Junie,
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            None,
        );
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
        assert!(snippet.contains("/usr/bin/rmcp-memex"));
        assert!(snippet.contains("--config"));
    }

    #[test]
    fn test_generate_json_snippet_includes_http_port_when_requested() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Standard(HostKind::Claude),
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            Some(8765),
        );
        assert!(snippet.contains("--http-port"));
        assert!(snippet.contains("8765"));
    }

    #[test]
    fn test_merge_json_config_empty() {
        let result = merge_json_config(
            "",
            &build_direct_host_entry(
                "/usr/bin/rmcp-memex",
                "~/.rmcp-servers/rmcp-memex/config.toml",
                None,
            ),
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["mcpServers"]["rmcp_memex"]["command"]
                .as_str()
                .unwrap()
                .contains("rmcp-memex")
        );
    }

    #[test]
    fn test_merge_json_config_preserves_http_port() {
        let result = merge_json_config(
            "",
            &build_direct_host_entry(
                "/usr/bin/rmcp-memex",
                "~/.rmcp-servers/rmcp-memex/config.toml",
                Some(8765),
            ),
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let args = parsed["mcpServers"]["rmcp_memex"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            vec![
                "serve",
                "--http-port",
                "8765",
                "--config",
                "~/.rmcp-servers/rmcp-memex/config.toml"
            ]
        );
    }

    #[test]
    fn test_merge_json_config_existing() {
        let existing = r#"{
  "mcpServers": {
    "other_server": {
      "command": "other",
      "args": []
    }
  }
        }"#;
        let result = merge_json_config(
            existing,
            &build_direct_host_entry(
                "/usr/bin/rmcp-memex",
                "~/.rmcp-servers/rmcp-memex/config.toml",
                None,
            ),
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Should preserve existing server
        assert!(
            parsed["mcpServers"]["other_server"]["command"]
                .as_str()
                .is_some()
        );
        // Should add rmcp_memex
        assert!(
            parsed["mcpServers"]["rmcp_memex"]["command"]
                .as_str()
                .unwrap()
                .contains("rmcp-memex")
        );
    }

    #[test]
    fn test_merge_toml_config_empty() {
        let result = merge_toml_config(
            "",
            &build_direct_host_entry(
                "/usr/bin/rmcp-memex",
                "~/.rmcp-servers/rmcp-memex/config.toml",
                None,
            ),
        )
        .unwrap();
        assert!(result.contains("[mcp_servers.rmcp_memex]"));
        assert!(result.contains("rmcp-memex"));
        assert!(result.contains("--config"));
    }

    #[test]
    fn test_merge_toml_config_existing() {
        let existing = r#"
[mcp_servers.other_server]
command = "other"
args = []
"#;
        let result = merge_toml_config(
            existing,
            &build_direct_host_entry(
                "/usr/bin/rmcp-memex",
                "~/.rmcp-servers/rmcp-memex/config.toml",
                None,
            ),
        )
        .unwrap();
        // Should preserve existing server
        assert!(result.contains("other_server"));
        // Should add rmcp_memex
        assert!(result.contains("rmcp-memex"));
        assert!(result.contains("--config"));
    }

    #[test]
    fn test_generate_mux_snippet_uses_proxy_command() {
        let snippet = generate_extended_snippet_mux(
            ExtendedHostKind::Standard(HostKind::Claude),
            "/custom/bin/rmcp-mux-proxy",
            DEFAULT_MUX_SOCKET_PATH,
        );
        assert!(snippet.contains("/custom/bin/rmcp-mux-proxy"));
        assert!(snippet.contains("--socket"));
        assert!(snippet.contains(DEFAULT_MUX_SOCKET_PATH));
    }

    #[test]
    fn test_build_mux_service_config_toml_uses_shared_daemon_shape() {
        let config = build_mux_service_config_toml(
            "/usr/bin/rmcp-memex",
            "~/.rmcp-servers/rmcp-memex/config.toml",
            Some(8765),
            4_194_304,
            "debug",
        )
        .unwrap();

        assert!(config.contains("[servers.rmcp-memex]"));
        assert!(config.contains("socket = \"~/.rmcp-servers/rmcp-memex/sockets/main.sock\""));
        assert!(config.contains("cmd = \"/usr/bin/rmcp-memex\""));
        assert!(config.contains("--http-port"));
        assert!(config.contains("8765"));
        assert!(config.contains("status_file = \"~/.rmcp-servers/rmcp-memex/status/main.json\""));
        assert!(config.contains("service_name = \"rmcp-memex\""));
        assert!(config.contains("max_request_bytes = 4194304"));
    }

    #[test]
    fn test_extended_host_kind_display_names() {
        assert_eq!(
            ExtendedHostKind::Standard(HostKind::Claude).label(),
            "Claude Desktop"
        );
        assert_eq!(ExtendedHostKind::ClaudeCode.label(), "Claude Code");
        assert_eq!(ExtendedHostKind::Junie.label(), "Junie");
    }
}
