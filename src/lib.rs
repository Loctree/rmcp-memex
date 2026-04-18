pub mod auth;
pub mod common;
pub mod embeddings;
pub mod engine;
pub mod handlers;
pub mod http;
pub mod mcp_core;
pub mod mcp_protocol;
mod mcp_runtime;
pub mod path_utils;
pub mod preprocessing;
pub mod query;
pub mod rag;
pub mod search;
pub mod security;
pub mod storage;
#[cfg(test)]
mod tests;
pub mod tools;

// CLI-only modules (require indicatif, ratatui, crossterm)
#[cfg(feature = "cli")]
pub mod progress;
#[cfg(feature = "cli")]
pub mod tui;

use anyhow::Result;
use tracing::Level;

// Re-export core types for library consumers
pub use auth::{
    AuthDenial, AuthManager, AuthResult, Scope, TokenEntry, TokenStoreFile, TokenStoreV2,
};
pub use embeddings::{
    DEFAULT_REQUIRED_DIMENSION, DimensionAdapter, EmbeddingClient, EmbeddingConfig, MLXBridge,
    MlxConfig, MlxMergeOptions, ProviderConfig, RerankerConfig, TokenConfig,
    cross_dimension_search_adapt, estimate_tokens, safe_chunk_size, truncate_to_token_limit,
    validate_batch_tokens, validate_chunk_tokens,
};
pub use handlers::{MCPServer, create_server};
pub use mcp_core::{
    McpCore, McpDispatch, McpTransport, shared_initialize_result, shared_tools_list_result,
};
pub use mcp_core::{dispatch_mcp_jsonrpc_request, dispatch_mcp_payload, dispatch_mcp_request};
pub use mcp_runtime::build_mcp_core;
pub use preprocessing::{
    IntegrityRecommendation, Message, PreprocessingConfig, PreprocessingStats, Preprocessor,
    TextIntegrityMetrics,
};
pub use query::{
    LoctreeSuggestion, QueryIntent, QueryRouter, RecommendedSearchMode, RoutingDecision,
    SearchModeRecommendation, TemporalHints, detect_intent,
};
pub use rag::{
    Chunk as PipelineChunk,
    ContextPrefixConfig,
    CrossStoreRecoveryBatchReport,
    CrossStoreRecoveryReport,
    CrossStoreRecoveryState,
    EmbeddedChunk,
    EnrichedChunk,
    FileContent,
    IndexResult,
    OnionSlice,
    OnionSliceConfig,
    PipelineConfig,
    PipelineEvent,
    PipelineGovernorConfig,
    PipelineResult,
    PipelineSnapshot,
    PipelineStats,
    RAGPipeline,
    SearchOptions,
    SearchResult,
    SliceLayer,
    SliceMode,
    compute_content_hash,
    create_enriched_chunks,
    create_onion_slices,
    inspect_cross_store_recovery,
    repair_cross_store_recovery,
    // Async pipeline exports
    run_pipeline,
};
pub use search::{
    BM25Config, BM25Index, HybridConfig, HybridSearchResult, HybridSearcher, SearchMode,
    StemLanguage,
};
#[allow(deprecated)]
pub use security::{NamespaceAccessManager, NamespaceSecurityConfig};
pub use storage::{
    ChromaDocument, CrossStoreRecoveryBatch, CrossStoreRecoveryDocumentRef,
    CrossStoreRecoveryStatus, GcConfig, GcStats, StorageManager, TableStats, parse_duration_string,
};

// High-level engine API
pub use engine::{BatchResult, MemexConfig, MemexEngine, MetaFilter, StoreItem};

// Canonical MCP metadata plus local Rust helper API.
pub use tools::{
    ToolDefinition, ToolResult, delete_document, delete_documents_by_filter, get_document,
    search_documents, store_document, store_documents_batch, tool_definitions,
};

// CLI-only re-exports (require indicatif, ratatui, crossterm)
#[cfg(feature = "cli")]
pub use progress::IndexProgressTracker;
#[cfg(feature = "cli")]
pub use tui::{
    CheckStatus, HealthCheckItem, HealthCheckResult, HealthChecker, HostDetection, HostKind,
    WizardConfig, detect_hosts, run_wizard,
};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Cache size in MB for moka in-memory cache
    pub cache_mb: usize,

    /// Path for embedded vector store (LanceDB)
    pub db_path: String,

    /// Max allowed request size (bytes) for JSON-RPC framing
    pub max_request_bytes: usize,

    /// Default log level to use when wiring tracing
    pub log_level: Level,

    /// Allowed paths for file access (whitelist).
    /// If empty, defaults to $HOME and current working directory.
    /// Supports ~ expansion and absolute paths.
    pub allowed_paths: Vec<String>,

    /// Namespace security configuration (token-based access control)
    pub security: NamespaceSecurityConfig,

    /// Embedding provider configuration (universal, config-driven)
    pub embeddings: EmbeddingConfig,

    /// Hybrid search configuration (vector + BM25)
    pub hybrid: HybridConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            cache_mb: 4096,
            db_path: "~/.rmcp-servers/rmcp-memex/lancedb".to_string(),
            max_request_bytes: 5 * 1024 * 1024,
            log_level: Level::INFO,
            allowed_paths: vec![],
            security: NamespaceSecurityConfig::default(),
            embeddings: EmbeddingConfig::default(),
            hybrid: HybridConfig::default(),
        }
    }
}

impl ServerConfig {
    #[doc(alias = "with_db_path")]
    pub fn with_storage_path(mut self, db_path: impl Into<String>) -> Self {
        self.db_path = db_path.into();
        self
    }

    #[deprecated(note = "use with_storage_path")]
    pub fn with_db_path(self, db_path: impl Into<String>) -> Self {
        self.with_storage_path(db_path)
    }
}

/// Helper to build and run the stdin/stdout server for library consumers.
pub async fn run_stdio_server(config: ServerConfig) -> Result<()> {
    let server = create_server(config).await?;
    server.run_stdio().await
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.cache_mb, 4096);
        assert_eq!(cfg.db_path, "~/.rmcp-servers/rmcp-memex/lancedb");
        assert_eq!(cfg.max_request_bytes, 5 * 1024 * 1024);
    }
}
