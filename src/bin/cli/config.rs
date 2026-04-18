use anyhow::Result;

use rmcp_memex::{
    DEFAULT_REQUIRED_DIMENSION, EmbeddingConfig, MlxConfig, ProviderConfig, RerankerConfig,
    path_utils,
};

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
pub fn load_or_discover_config(
    explicit_path: Option<&str>,
) -> Result<(FileConfig, Option<String>)> {
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

#[derive(serde::Deserialize, Default, Clone)]
pub struct FileConfig {
    /// Legacy compatibility field. Parsed but ignored when building ServerConfig.
    pub mode: Option<String>,
    /// Legacy compatibility field. Parsed but ignored when building ServerConfig.
    pub features: Option<String>,
    pub cache_mb: Option<usize>,
    pub db_path: Option<String>,
    pub max_request_bytes: Option<usize>,
    pub log_level: Option<String>,
    pub allowed_paths: Option<Vec<String>>,
    /// Enable namespace token-based access control
    pub security_enabled: Option<bool>,
    /// Path to token store file
    pub token_store_path: Option<String>,
    /// Enable preprocessing to filter noise from documents before indexing
    pub preprocessing_enabled: Option<bool>,
    /// New: universal embedding config with provider cascade
    #[serde(default)]
    pub embeddings: Option<EmbeddingsFileConfig>,
    /// Legacy: MLX embedding server configuration (deprecated)
    #[serde(default)]
    pub mlx: Option<MlxFileConfig>,
    /// Automatic maintenance configuration
    #[serde(default)]
    pub maintenance: Option<MaintenanceFileConfig>,
    /// Bearer token for HTTP auth (mutating endpoints)
    pub auth_token: Option<String>,
    /// Bind address for HTTP server (default: 127.0.0.1)
    pub bind_address: Option<String>,
    /// Allowed CORS origins (comma-separated list)
    pub cors_origins: Option<String>,
    /// Auth mode: "mutating-only", "all-routes", or "namespace-acl"
    pub auth_mode: Option<String>,
    /// Allow ?token= query parameter on read GETs
    pub allow_query_token: Option<bool>,
}

/// New embedding configuration from TOML
#[derive(serde::Deserialize, Clone)]
pub struct EmbeddingsFileConfig {
    #[serde(default = "default_dimension")]
    pub required_dimension: usize,
    #[serde(default)]
    pub providers: Vec<ProviderFileConfig>,
    #[serde(default)]
    pub reranker: Option<RerankerFileConfig>,
}

pub fn default_dimension() -> usize {
    DEFAULT_REQUIRED_DIMENSION
}

impl Default for EmbeddingsFileConfig {
    fn default() -> Self {
        Self {
            required_dimension: default_dimension(),
            providers: vec![],
            reranker: None,
        }
    }
}

#[derive(serde::Deserialize, Clone)]
pub struct ProviderFileConfig {
    pub name: String,
    pub base_url: String,
    pub model: String,
    #[serde(default = "default_priority")]
    pub priority: u8,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
}

pub fn default_priority() -> u8 {
    10
}

pub fn default_endpoint() -> String {
    "/v1/embeddings".to_string()
}

#[derive(serde::Deserialize, Clone)]
pub struct RerankerFileConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default = "default_rerank_endpoint")]
    pub endpoint: String,
}

pub fn default_rerank_endpoint() -> String {
    "/v1/rerank".to_string()
}

/// Legacy MLX embedding server configuration from TOML (deprecated)
#[derive(serde::Deserialize, Default, Clone)]
pub struct MlxFileConfig {
    #[serde(default)]
    pub disabled: bool,
    pub local_port: Option<u16>,
    pub dragon_url: Option<String>,
    pub dragon_port: Option<u16>,
    pub embedder_model: Option<String>,
    pub reranker_model: Option<String>,
    pub reranker_port_offset: Option<u16>,
}

impl MlxFileConfig {
    /// Convert legacy config to MlxConfig for backward compat
    pub fn to_mlx_config(&self) -> MlxConfig {
        let mut config = MlxConfig::from_env();
        config.merge_file_config(rmcp_memex::MlxMergeOptions {
            disabled: Some(self.disabled),
            local_port: self.local_port,
            dragon_url: self.dragon_url.clone(),
            dragon_port: self.dragon_port,
            embedder_model: self.embedder_model.clone(),
            reranker_model: self.reranker_model.clone(),
            reranker_port_offset: self.reranker_port_offset,
        });
        config
    }
}

/// Maintenance configuration for automatic optimization
#[derive(serde::Deserialize, Default, Clone)]
pub struct MaintenanceFileConfig {
    /// Enable automatic optimization when version threshold is exceeded
    #[serde(default)]
    pub auto_optimize: bool,

    /// Number of versions that triggers automatic optimization (default: 50)
    #[serde(default = "default_version_threshold")]
    pub version_threshold: usize,

    /// Automatically cleanup versions older than N days (optional)
    #[serde(default)]
    pub auto_cleanup_days: Option<u64>,
}

pub fn default_version_threshold() -> usize {
    50
}

impl FileConfig {
    /// Convert to EmbeddingConfig - new format takes precedence over legacy
    pub fn resolve_embedding_config(&self) -> EmbeddingConfig {
        // New format takes precedence, even when it only overrides dimension or reranker.
        if let Some(ref emb) = self.embeddings {
            let mut config = if emb.providers.is_empty() {
                if let Some(ref mlx) = self.mlx {
                    tracing::warn!(
                        "Using legacy [mlx] providers with [embeddings] overrides - please migrate to [embeddings.providers]"
                    );
                    mlx.to_mlx_config().to_embedding_config()
                } else {
                    MlxConfig::from_env().to_embedding_config()
                }
            } else {
                EmbeddingConfig::default()
            };

            config.required_dimension = emb.required_dimension;

            if !emb.providers.is_empty() {
                config.providers = emb
                    .providers
                    .iter()
                    .map(|p| ProviderConfig {
                        name: p.name.clone(),
                        base_url: p.base_url.clone(),
                        model: p.model.clone(),
                        priority: p.priority,
                        endpoint: p.endpoint.clone(),
                    })
                    .collect();
            }

            if let Some(ref reranker) = emb.reranker {
                config.reranker = RerankerConfig {
                    base_url: Some(reranker.base_url.clone()),
                    model: Some(reranker.model.clone()),
                    endpoint: reranker.endpoint.clone(),
                };
            }

            return config;
        }

        // Fallback to legacy [mlx] config
        if let Some(ref mlx) = self.mlx {
            tracing::warn!("Using legacy [mlx] config - please migrate to [embeddings.providers]");
            return mlx.to_mlx_config().to_embedding_config();
        }

        // Use defaults from environment variables (legacy support)
        MlxConfig::from_env().to_embedding_config()
    }
}

pub struct ResolvedConfig {
    pub file_cfg: FileConfig,
    pub config_path: Option<String>,
    pub db_path: String,
    pub embedding_config: rmcp_memex::EmbeddingConfig,
    pub maintenance_config: Option<MaintenanceFileConfig>,
}

impl ResolvedConfig {
    pub fn load(cli_config_path: Option<&str>, cli_db_path: Option<&str>) -> anyhow::Result<Self> {
        let (file_cfg, config_path) = load_or_discover_config(cli_config_path)?;
        if let Some(ref path) = config_path {
            eprintln!("Using config: {}", path);
        }

        let embedding_config = file_cfg.resolve_embedding_config();
        let maintenance_config = file_cfg.maintenance.clone();

        let db_path = cli_db_path
            .map(|s| s.to_string())
            .or_else(|| file_cfg.db_path.clone())
            .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
        let db_path = shellexpand::tilde(&db_path).to_string();

        Ok(Self {
            file_cfg,
            config_path,
            db_path,
            embedding_config,
            maintenance_config,
        })
    }
}
