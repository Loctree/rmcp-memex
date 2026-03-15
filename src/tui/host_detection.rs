//! Host detection module for MCP server configurations.
//!
//! Scans known locations for MCP host configurations (Codex, Cursor, Claude Desktop, JetBrains).
//! Also provides config writing functionality for the wizard.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// HOST TYPES (previously from rmcp-common)
// =============================================================================

/// Supported MCP host application kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostKind {
    /// Codex CLI (uses TOML config)
    Codex,
    /// Cursor editor (uses JSON config)
    Cursor,
    /// VS Code with MCP extension (uses JSON config)
    VSCode,
    /// Claude Desktop application (uses JSON config)
    Claude,
    /// JetBrains IDEs with MCP plugin (uses JSON config)
    JetBrains,
    /// Unknown or custom host
    Unknown,
}

impl HostKind {
    /// Returns a lowercase label for the host kind.
    pub fn as_label(&self) -> &'static str {
        match self {
            HostKind::Codex => "codex",
            HostKind::Cursor => "cursor",
            HostKind::VSCode => "vscode",
            HostKind::Claude => "claude",
            HostKind::JetBrains => "jetbrains",
            HostKind::Unknown => "unknown",
        }
    }

    /// Returns a human-readable display name for the host kind.
    pub fn display_name(&self) -> &'static str {
        match self {
            HostKind::Codex => "Codex CLI",
            HostKind::Cursor => "Cursor",
            HostKind::VSCode => "VS Code",
            HostKind::Claude => "Claude Desktop",
            HostKind::JetBrains => "JetBrains IDEs",
            HostKind::Unknown => "Unknown",
        }
    }
}

/// Configuration file format for MCP hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostFormat {
    /// TOML format (used by Codex)
    Toml,
    /// JSON format (used by most other hosts)
    Json,
}

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
            "Detected (no rmcp_memex)"
        }
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
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
    pub fn display_name(&self) -> &'static str {
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
            let has_rmcp = servers
                .iter()
                .any(|s| s.name.contains("rmcp_memex") || s.command.contains("rmcp_memex"));
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

/// Generate a config snippet for an extended host kind.
pub fn generate_extended_snippet(
    kind: ExtendedHostKind,
    binary_path: &str,
    db_path: &str,
) -> String {
    match get_extended_host_config_path(kind) {
        Some((_, HostFormat::Toml)) => {
            format!(
                r#"[mcp_servers.rmcp_memex]
command = "{}"
args = ["serve", "--db-path", "{}", "--log-level", "info"]
"#,
                binary_path, db_path
            )
        }
        Some((_, HostFormat::Json)) => {
            format!(
                r#"{{
  "mcpServers": {{
    "rmcp_memex": {{
      "command": "{}",
      "args": ["serve", "--db-path", "{}", "--log-level", "info"]
    }}
  }}
}}"#,
                binary_path, db_path
            )
        }
        None => String::new(),
    }
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

/// Merge rmcp_memex server entry into existing JSON config
fn merge_json_config(existing_content: &str, binary_path: &str, db_path: &str) -> Result<String> {
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
    config["mcpServers"]["rmcp_memex"] = serde_json::json!({
        "command": binary_path,
        "args": ["serve", "--db-path", db_path, "--log-level", "info"],
        "description": "RAG memory with vector search"
    });

    serde_json::to_string_pretty(&config).with_context(|| "Failed to serialize JSON config")
}

/// Merge rmcp_memex server entry into existing TOML config
fn merge_toml_config(existing_content: &str, binary_path: &str, db_path: &str) -> Result<String> {
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
        let mut entry = toml::map::Map::new();
        entry.insert(
            "command".to_string(),
            toml::Value::String(binary_path.to_string()),
        );
        entry.insert(
            "args".to_string(),
            toml::Value::Array(vec![
                toml::Value::String("serve".to_string()),
                toml::Value::String("--db-path".to_string()),
                toml::Value::String(db_path.to_string()),
                toml::Value::String("--log-level".to_string()),
                toml::Value::String("info".to_string()),
            ]),
        );
        mcp_servers.insert("rmcp_memex".to_string(), toml::Value::Table(entry));
    }

    Ok(toml::to_string_pretty(&config)?)
}

/// Write host config, merging with existing config if present.
/// Creates a backup before modifying existing files.
///
/// # Arguments
/// * `host` - The detected host to write config for
/// * `binary_path` - Path to the rmcp_memex binary
/// * `db_path` - Path to the LanceDB database
///
/// # Returns
/// * `Ok(WriteResult)` with details about the write operation
/// * `Err` if the write fails
pub fn write_host_config(
    host: &HostDetection,
    binary_path: &str,
    db_path: &str,
) -> Result<WriteResult> {
    let host_name = host.kind.display_name().to_string();

    // Ensure parent directory exists
    if let Some(parent) = host.path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    // Create backup if file exists
    let backup_path = if host.exists {
        Some(create_backup(&host.path)?)
    } else {
        None
    };

    use crate::path_utils::validate_write_path;

    // Read existing content or use empty string
    let existing_content = if host.exists {
        // Validate path before reading
        let (_safe_path, content) =
            crate::path_utils::safe_read_to_string(&host.path.to_string_lossy())
                .with_context(|| format!("Cannot read config: {}", host.path.display()))?;
        content
    } else {
        String::new()
    };

    // Merge config based on format
    let new_content = match host.format {
        HostFormat::Json => merge_json_config(&existing_content, binary_path, db_path)?,
        HostFormat::Toml => merge_toml_config(&existing_content, binary_path, db_path)?,
    };

    // Validate path before writing
    let safe_write_path = validate_write_path(&host.path).with_context(|| {
        format!(
            "Cannot write config: path validation failed for {}",
            host.path.display()
        )
    })?;

    // Write the merged config
    std::fs::write(&safe_write_path, &new_content)
        .with_context(|| format!("Failed to write config to {}", safe_write_path.display()))?;

    Ok(WriteResult {
        host_name,
        config_path: host.path.clone(),
        backup_path,
        created: !host.exists,
    })
}

/// Write config for an extended host kind (including ClaudeCode and Junie)
pub fn write_extended_host_config(
    kind: ExtendedHostKind,
    binary_path: &str,
    db_path: &str,
) -> Result<WriteResult> {
    let (path, format) =
        get_extended_host_config_path(kind).ok_or_else(|| anyhow::anyhow!("Unknown host kind"))?;

    let exists = path.exists();
    let host = HostDetection {
        kind: match kind {
            ExtendedHostKind::Standard(k) => k,
            _ => HostKind::Unknown, // Use Unknown for extended types
        },
        path: path.clone(),
        format,
        exists,
        has_rmcp_memex: false,
        servers: Vec::new(),
    };

    let mut result = write_host_config(&host, binary_path, db_path)?;
    result.host_name = kind.display_name().to_string();
    Ok(result)
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
                    let has_rmcp = servers
                        .iter()
                        .any(|s| s.name.contains("rmcp_memex") || s.command.contains("rmcp_memex"));
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
    fn test_generate_toml_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Standard(HostKind::Codex),
            "/usr/bin/rmcp_memex",
            "~/.rmcp/db",
        );
        assert!(snippet.contains("[mcp_servers.rmcp_memex]"));
        assert!(snippet.contains("/usr/bin/rmcp_memex"));
    }

    #[test]
    fn test_generate_json_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::Standard(HostKind::Claude),
            "/usr/bin/rmcp_memex",
            "~/.rmcp/db",
        );
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
    }

    #[test]
    fn test_generate_extended_claude_code_snippet() {
        let snippet = generate_extended_snippet(
            ExtendedHostKind::ClaudeCode,
            "/usr/bin/rmcp_memex",
            "~/.rmcp/db",
        );
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
    }

    #[test]
    fn test_generate_extended_junie_snippet() {
        let snippet =
            generate_extended_snippet(ExtendedHostKind::Junie, "/usr/bin/rmcp_memex", "~/.rmcp/db");
        assert!(snippet.contains("\"mcpServers\""));
        assert!(snippet.contains("\"rmcp_memex\""));
    }

    #[test]
    fn test_merge_json_config_empty() {
        let result = merge_json_config("", "/usr/bin/rmcp_memex", "~/.rmcp/db").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["mcpServers"]["rmcp_memex"]["command"]
                .as_str()
                .unwrap()
                .contains("rmcp_memex")
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
        let result = merge_json_config(existing, "/usr/bin/rmcp_memex", "~/.rmcp/db").unwrap();
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
                .contains("rmcp_memex")
        );
    }

    #[test]
    fn test_merge_toml_config_empty() {
        let result = merge_toml_config("", "/usr/bin/rmcp_memex", "~/.rmcp/db").unwrap();
        assert!(result.contains("[mcp_servers.rmcp_memex]"));
        assert!(result.contains("rmcp_memex"));
    }

    #[test]
    fn test_merge_toml_config_existing() {
        let existing = r#"
[mcp_servers.other_server]
command = "other"
args = []
"#;
        let result = merge_toml_config(existing, "/usr/bin/rmcp_memex", "~/.rmcp/db").unwrap();
        // Should preserve existing server
        assert!(result.contains("other_server"));
        // Should add rmcp_memex
        assert!(result.contains("rmcp_memex"));
    }

    #[test]
    fn test_extended_host_kind_display_names() {
        assert_eq!(
            ExtendedHostKind::Standard(HostKind::Claude).display_name(),
            "Claude Desktop"
        );
        assert_eq!(ExtendedHostKind::ClaudeCode.display_name(), "Claude Code");
        assert_eq!(ExtendedHostKind::Junie.display_name(), "Junie");
    }
}
