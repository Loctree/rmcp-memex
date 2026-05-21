use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::Level;
use walkdir::WalkDir;

use rmcp_memex::{NamespaceSecurityConfig, ServerConfig, path_utils};

pub const DEFAULT_DASHBOARD_PORT: u16 = 8987;
pub const DEFAULT_SSE_PORT: u16 = 8997;
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
#[derive(Parser, Debug)]
#[command(
    name = "rmcp-memex",
    bin_name = "rmcp-memex",
    author,
    version,
    about = "rmcp-memex: custom Rust MCP kernel for RAG and long-term memory.\nCanonical entrypoint for stdio (native MCP) and HTTP/SSE (multi-agent) transports.",
    long_about = "rmcp-memex is a custom Rust MCP kernel providing RAG and long-term memory capabilities to AI agents via LanceDB.\n\nIt exposes two explicit transport modes from a single canonical surface:\n1. stdio (Standard MCP): Native MCP integration for local agents.\n2. HTTP/SSE (Multi-Agent Daemon): Central daemon mode allowing concurrent AI agents to access the same memory pool over the network.\n\nrmcp-memex is the only supported binary name. The GitHub installer may also create rmcp_memex as a legacy compatibility symlink for older scripts."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Optional config file (TOML) to load settings from; CLI flags override file when set.
    #[arg(long, global = true)]
    pub config: Option<String>,

    /// Legacy compatibility shim. Ignored at runtime.
    #[arg(long, value_parser = ["memory", "full"], global = true, hide = true)]
    pub mode: Option<String>,

    /// Legacy compatibility shim. Ignored at runtime.
    #[arg(long, global = true, hide = true)]
    pub features: Option<String>,

    /// Cache size in MB
    #[arg(long, global = true)]
    pub cache_mb: Option<usize>,

    /// Path for embedded vector store (LanceDB)
    #[arg(long, global = true)]
    pub db_path: Option<String>,

    /// Max allowed request size in bytes for JSON-RPC framing
    #[arg(long, global = true)]
    pub max_request_bytes: Option<usize>,

    /// Log level
    #[arg(long, global = true)]
    pub log_level: Option<String>,

    /// Allowed paths for file access (whitelist). Can be specified multiple times.
    /// If not set, defaults to $HOME and current working directory.
    /// Supports ~ expansion and absolute paths.
    #[arg(long, global = true, action = clap::ArgAction::Append)]
    pub allowed_paths: Option<Vec<String>>,

    /// Enable namespace token-based access control.
    /// When enabled, protected namespaces require a token for access.
    #[arg(long, global = true)]
    pub security_enabled: bool,

    /// Path to token store file for namespace access tokens.
    /// Defaults to ~/.rmcp-servers/rmcp-memex/tokens.json when security is enabled.
    #[arg(long, global = true)]
    pub token_store_path: Option<String>,

    /// HTTP/SSE server port for multi-agent access.
    /// When set, starts an HTTP server alongside MCP stdio.
    /// Agents can query via HTTP instead of holding LanceDB lock directly.
    /// Example: --http-port 8997
    #[arg(long, global = true)]
    pub http_port: Option<u16>,

    /// Run HTTP server only, without MCP stdio.
    /// Use this for daemon mode where agents connect via HTTP.
    /// Requires --http-port to be set.
    #[arg(long, global = true)]
    pub http_only: bool,

    /// Bearer token for authenticating mutating HTTP endpoints.
    /// Can also be set via MEMEX_AUTH_TOKEN env var.
    #[arg(long, global = true)]
    pub auth_token: Option<String>,

    /// Bind address for the HTTP server. Defaults to 127.0.0.1 (localhost only).
    /// Use 0.0.0.0 to expose on all interfaces (requires --auth-token for safety).
    #[arg(long, global = true)]
    pub bind_address: Option<String>,

    /// Allowed CORS origins (comma-separated). If empty, defaults to same-origin
    /// when bound to non-localhost, or permissive when bound to localhost.
    #[arg(long, global = true)]
    pub cors_origins: Option<String>,

    /// Allow binding to non-loopback addresses without --auth-token.
    /// By default, binding to e.g. 0.0.0.0 without auth is a hard error.
    /// This flag downgrades it to a warning.
    #[arg(long, global = true)]
    pub allow_network_without_auth: bool,

    /// Auth enforcement mode for HTTP endpoints.
    /// - mutating-only (default): bearer required only on mutating + MCP routes
    /// - all-routes: bearer required on ALL routes
    /// - namespace-acl: reserved for Track C (namespace-level ACL)
    #[arg(long, global = true, default_value = "mutating-only",
           value_parser = ["mutating-only", "all-routes", "namespace-acl"])]
    pub auth_mode: String,

    /// Allow passing bearer token as ?token= query parameter on read GET endpoints.
    /// Disabled by default. Only effective when --auth-mode is all-routes.
    #[arg(long, global = true)]
    pub allow_query_token: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run the MCP server (default if no subcommand specified)
    Serve,

    /// Run the local dashboard server and open it in the default browser.
    Dashboard {
        /// Dashboard HTTP port (default: 8987)
        #[arg(long, short = 'p')]
        port: Option<u16>,

        /// Do not open the dashboard in a browser after startup
        #[arg(long)]
        no_open: bool,
    },

    /// Run the HTTP/SSE daemon on the agent-facing port.
    Sse {
        /// HTTP/SSE port (default: 8997)
        #[arg(long, short = 'p')]
        port: Option<u16>,
    },

    /// Launch interactive configuration wizard
    #[command(alias = "config")]
    Wizard {
        /// Dry run mode - show changes without writing files
        #[arg(long)]
        dry_run: bool,
    },

    /// Quick stats and health check for namespaces
    ///
    /// Shows chunk count, date range, top topics, and storage info.
    ///
    /// Examples:
    ///   rmcp-memex overview           # All namespaces
    ///   rmcp-memex overview memories  # Specific namespace
    Overview {
        /// Namespace to get overview for (optional, shows all if not specified)
        namespace: Option<String>,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Deep exploration with all details - drill into onion layers
    ///
    /// Shows ALL onion layers (outer/middle/inner/core), both BM25 and vector scores,
    /// full metadata, and related chunks.
    ///
    /// Examples:
    ///   rmcp-memex dive -n memories -q "dragon"
    ///   rmcp-memex dive -n memories -q "dragon" --verbose
    Dive {
        /// Namespace to search in
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results per layer
        #[arg(long, short = 'l', default_value = "5")]
        limit: usize,

        /// Show extra verbose output (full text, all metadata)
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Batch index documents into vector store
    Index {
        /// Path to file or directory to index
        #[arg(required = true)]
        path: PathBuf,

        /// Namespace for indexed documents (default: "rag")
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Recursively walk subdirectories
        #[arg(long, short = 'r')]
        recursive: bool,

        /// Glob pattern to filter files (e.g. "*.md", "*.pdf")
        #[arg(long, short = 'g')]
        glob: Option<String>,

        /// Maximum depth when walking directories (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_depth: usize,

        /// Enable preprocessing to filter noise (tool artifacts, CLI output)
        /// before indexing. Reduces vector storage size and improves search quality.
        /// Note: timestamps are preserved by default; use --sanitize-metadata to remove them.
        #[arg(long, short = 'p')]
        preprocess: bool,

        /// Sanitize timestamps, UUIDs, and session IDs from content.
        /// By default, these are preserved for temporal queries.
        /// Use this flag when you want to anonymize or normalize the data.
        #[arg(long)]
        sanitize_metadata: bool,

        /// Slicing mode for document chunking:
        /// - "onion" (default): Hierarchical slices (outer/middle/inner/core) for efficient context
        /// - "onion-fast" / "fast": Only outer+core layers (2x faster, good for large datasets)
        /// - "flat": Traditional fixed-size chunks with overlap
        #[arg(long, short = 's', default_value = "onion", value_parser = ["onion", "onion-fast", "fast", "flat"])]
        slice_mode: String,

        /// Enable exact-match deduplication (default: enabled).
        /// Skips indexing files whose content already exists in the namespace.
        /// Uses SHA256 hash of original content before any preprocessing.
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        dedup: bool,

        /// Show progress bar with ETA when running in an interactive terminal.
        /// Non-interactive runs fall back to line logs.
        #[arg(long)]
        progress: bool,

        /// Resume from last checkpoint if interrupted.
        /// Saves progress after each committed file to .index-checkpoint-<namespace>.json.
        /// On restart, skips already indexed files and continues.
        #[arg(long)]
        resume: bool,

        /// Enable async pipeline mode for concurrent indexing.
        /// Runs file reading, chunking, embedding, and storage in parallel
        /// using tokio channels. Can significantly speed up large batch operations.
        /// Supports live progress output and commit-based resume checkpoints.
        #[arg(long)]
        pipeline: bool,

        /// Maximum number of embedding requests to keep in flight in pipeline mode.
        /// With --pipeline-governor disabled this is a fixed concurrency limit.
        /// With --pipeline-governor enabled this becomes the governor's ceiling.
        #[arg(long, default_value = "1", value_parser = clap::value_parser!(u8).range(1..=8))]
        pipeline_embed_concurrency: u8,

        /// Enable adaptive pipeline flow control for embedding batch sizes and concurrency.
        /// Uses embed latency and queue pressure to increase slowly and back off quickly.
        #[arg(long)]
        pipeline_governor: bool,

        /// Number of files to process in parallel (default: 4, max: 16).
        /// Higher values can speed up indexing on multi-core systems,
        /// but may increase memory usage and API pressure.
        /// Note: This is ignored when --pipeline is enabled.
        #[arg(long, short = 'P', default_value = "4", value_parser = clap::value_parser!(u8).range(1..=16))]
        parallel: u8,
    },

    /// Smart semantic search within a namespace
    ///
    /// Finds relevant information using vector similarity search with intelligent
    /// defaults. Results include relevance scores, timestamps, and metadata.
    ///
    /// Examples:
    ///   rmcp-memex search -n memories -q "when did we buy dragon"
    ///   rmcp-memex search -n memories -q "dragon" --deep
    ///   rmcp-memex search -n memories -q "dragon" -l 20
    ///   rmcp-memex search -n memories -q "dragon" --mode hybrid
    Search {
        /// Namespace to search in
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results to return (default: 10)
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,

        /// Deep search: include all layers (outer/middle/inner/core) instead of just outer
        #[arg(long)]
        deep: bool,

        /// Filter by specific layer (outer, middle, inner, core)
        #[arg(long, value_parser = ["outer", "middle", "inner", "core"])]
        layer: Option<String>,

        /// Search mode: vector (similarity only), keyword/bm25 (lexical only), or hybrid (default)
        /// Hybrid combines vector and BM25 using score fusion for best results.
        #[arg(long, short = 'm', default_value = "hybrid", value_parser = ["vector", "keyword", "bm25", "hybrid"])]
        mode: String,

        /// Auto-detect query intent and select optimal search mode.
        /// Overrides --mode when enabled. Uses QueryRouter to analyze query.
        #[arg(long)]
        auto_route: bool,

        /// Show relevance scores prominently (enabled by default)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        scores: bool,
    },

    /// Expand a slice to get its children (drill down in onion hierarchy)
    Expand {
        /// Namespace containing the slice
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Slice ID to expand
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Get a specific chunk by namespace and ID
    Get {
        /// Namespace containing the chunk
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Chunk ID to retrieve
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Output result as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// RAG search across all namespaces or a specific one
    RagSearch {
        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results to return
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,

        /// Optional namespace to limit search to
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// List all namespaces with optional statistics
    Namespaces {
        /// Show statistics (document count, etc.)
        #[arg(long, short = 's')]
        stats: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Export a namespace to JSONL file for portable backup
    ///
    /// Each document is written as a JSON line with: id, text, metadata, content_hash,
    /// and optionally embeddings. Use with 'import' command for backup/restore.
    ///
    /// Examples:
    ///   rmcp-memex export -n memories -o backup.jsonl
    ///   rmcp-memex export -n memories --include-embeddings -o full-backup.jsonl
    Export {
        /// Namespace to export
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Output file path (.jsonl format, stdout if not specified)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Include vector embeddings in export (makes files much larger)
        #[arg(long)]
        include_embeddings: bool,

        /// Database path override
        #[arg(long)]
        db_path: Option<String>,
    },

    /// Upsert a text chunk directly into vector memory (for hooks/scripts)
    Upsert {
        /// Namespace for the chunk
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Unique ID for the chunk
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Text content (if not provided, reads from stdin)
        #[arg(long, short = 't')]
        text: Option<String>,

        /// Optional metadata as JSON string
        #[arg(long, short = 'm', default_value = "{}")]
        metadata: String,
    },

    /// Optimize database: compact files and cleanup old versions
    ///
    /// Runs both compaction (merge small files) and pruning (remove old versions).
    /// Use this after large indexing operations to improve query performance
    /// and reduce file descriptor usage.
    Optimize,

    /// Show database health status and recommendations
    ///
    /// Checks database connectivity, embedder availability, namespace stats,
    /// and provides maintenance recommendations.
    ///
    /// Examples:
    ///   rmcp-memex health            # Full health check
    ///   rmcp-memex health --quick    # Skip embedder check (faster)
    ///   rmcp-memex health --json     # JSON output for scripting
    Health {
        /// Skip embedder connectivity check (faster, DB-only)
        #[arg(long, short = 'q')]
        quick: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Recall memories about a topic with synthesized summary
    ///
    /// Searches your memories and presents results as a coherent summary,
    /// using the onion slice architecture (outer layers = summaries).
    ///
    /// Examples:
    ///   rmcp-memex recall "Vista architecture"          # Search all namespaces
    ///   rmcp-memex recall "dragon setup" -n memories    # Specific namespace
    ///   rmcp-memex recall "auth flow" --limit 20        # More sources
    Recall {
        /// What to recall (search query)
        query: String,

        /// Limit to specific namespace (default: search all)
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Maximum number of sources to consider (default: 10)
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Show timeline of indexed content
    ///
    /// Displays when documents were indexed, grouped by month.
    /// Useful for understanding temporal coverage of your memory.
    ///
    /// Examples:
    ///   rmcp-memex timeline                           # All namespaces
    ///   rmcp-memex timeline -n memories               # Specific namespace
    ///   rmcp-memex timeline -n memories --since 30d   # Last 30 days
    ///   rmcp-memex timeline --gaps                    # Show only gaps
    Timeline {
        /// Filter to specific namespace (default: all namespaces)
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Show entries since this time (e.g., "30d", "2025-01", "2024-12-01")
        #[arg(long)]
        since: Option<String>,

        /// Only show gaps in the timeline (days with no indexed content)
        #[arg(long)]
        gaps: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Compact database files into larger chunks
    ///
    /// Merges small data files into larger ones for better read performance.
    /// Run this after many small inserts to reduce "too many open files" errors.
    Compact,

    /// Cleanup old database versions
    ///
    /// Removes old versions of the data that are no longer needed.
    /// By default, keeps versions from the last 7 days.
    Cleanup {
        /// Remove versions older than N days (default: 7)
        #[arg(long, default_value = "7")]
        older_than_days: u64,
    },

    /// Show database statistics
    ///
    /// Displays row count, version count, and storage information.
    Stats,

    /// Garbage collection: clean up orphaned data
    ///
    /// Removes orphan embeddings, empty namespaces, and old documents.
    /// Always runs in dry-run mode unless you pass the --execute flag.
    ///
    /// Examples:
    ///   rmcp-memex gc --remove-orphans                    # Dry run: show orphans
    ///   rmcp-memex gc --remove-orphans --execute          # Actually remove orphans
    ///   rmcp-memex gc --older-than 90d                    # Dry run: docs older than 90 days
    ///   rmcp-memex gc --older-than 6m --namespace logs    # Only in 'logs' namespace
    ///   rmcp-memex gc --remove-orphans --remove-empty --older-than 1y --execute
    Gc {
        /// Remove orphan embeddings (documents with parent_id pointing to non-existent documents)
        #[arg(long)]
        remove_orphans: bool,

        /// Remove empty namespaces (report namespaces with 0 documents)
        #[arg(long)]
        remove_empty: bool,

        /// Remove documents older than this duration (e.g., "30d", "6m", "1y")
        #[arg(long)]
        older_than: Option<String>,

        /// Actually execute the cleanup (default is dry-run mode)
        #[arg(long)]
        execute: bool,

        /// Limit to specific namespace (optional, applies to all if not set)
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Inspect or repair pending Lance/BM25 recovery ledgers
    ///
    /// This is the explicit recovery contract for partial cross-store writes.
    /// It does not claim crash-safe atomicity. Instead it inspects persisted
    /// batch ledgers, reports divergence, and can replay BM25 writes to match
    /// current Lance truth.
    ///
    /// Examples:
    ///   rmcp-memex repair-writes
    ///   rmcp-memex repair-writes --execute
    ///   rmcp-memex repair-writes -n memories --json
    RepairWrites {
        /// Limit inspection/repair to a single namespace
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Actually execute reconciliation. Default is dry-run/report-only.
        #[arg(long)]
        execute: bool,

        /// Output results as JSON instead of human-readable text
        #[arg(long)]
        json: bool,
    },

    /// Search across all namespaces
    ///
    /// Performs a unified search across every namespace, merging and ranking results.
    ///
    /// Examples:
    ///   rmcp-memex cross-search "error handling"
    ///   rmcp-memex cross-search "config" --mode hybrid --limit 5 --total-limit 20
    ///   rmcp-memex cross-search "memory leak" --json
    CrossSearch {
        /// The search query
        query: String,

        /// Maximum results per namespace (default: 10)
        #[arg(long, default_value = "10")]
        limit: usize,

        /// Maximum total results after merging (default: 50)
        #[arg(long, default_value = "50")]
        total_limit: usize,

        /// Search mode: vector, bm25/keyword, or hybrid (default: hybrid)
        #[arg(long, default_value = "hybrid")]
        mode: String,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Merge multiple LanceDB databases into one with deduplication
    ///
    /// Combines documents from multiple source databases into a single target database.
    /// Useful for consolidating memory across machines or instances.
    ///
    /// Examples:
    ///   rmcp-memex merge --source ~/db1 --source ~/db2 --target ~/merged
    ///   rmcp-memex merge --source ~/db1 --source ~/db2 --target ~/merged --dedup
    ///   rmcp-memex merge --source ~/dragon-db --target ~/merged --namespace-prefix "dragon:"
    ///   rmcp-memex merge --source ~/db1 --target ~/merged --dry-run
    Merge {
        /// Source database paths (can specify multiple times)
        #[arg(long, short = 's', required = true, action = clap::ArgAction::Append)]
        source: Vec<PathBuf>,

        /// Target database path (will be created if not exists)
        #[arg(long, short = 't', required = true)]
        target: PathBuf,

        /// Deduplicate by content_hash (skip documents with same hash)
        #[arg(long, short = 'd')]
        dedup: bool,

        /// Prefix to add to source namespaces (e.g., "dragon:" -> "dragon:memories")
        #[arg(long, short = 'p')]
        namespace_prefix: Option<String>,

        /// Show what would be merged without actually doing it
        #[arg(long)]
        dry_run: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Find and remove duplicate documents based on content hash
    ///
    /// Groups documents by content_hash and removes duplicates, keeping one
    /// document per unique content based on the --keep strategy.
    ///
    /// Examples:
    ///   rmcp-memex dedup                          # All namespaces, dry-run
    ///   rmcp-memex dedup -n memories              # Specific namespace
    ///   rmcp-memex dedup --dry-run false          # Actually remove duplicates
    ///   rmcp-memex dedup --keep newest            # Keep newest duplicates
    ///   rmcp-memex dedup --cross-namespace        # Dedup across all namespaces
    Dedup {
        /// Specific namespace to deduplicate (if not set, processes all namespaces separately)
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Show duplicates without removing them (default: true)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        dry_run: bool,

        /// Strategy for which document to keep when duplicates are found:
        /// - "oldest": Keep the document with the earliest ID (lexicographic, default)
        /// - "newest": Keep the document with the latest ID (lexicographic)
        /// - "highest-score": Keep the document that appears first in vector search
        #[arg(long, default_value = "oldest", value_parser = ["oldest", "newest", "highest-score"])]
        keep: String,

        /// Deduplicate across all namespaces (treat entire DB as one pool).
        /// By default, deduplication is done within each namespace separately.
        #[arg(long)]
        cross_namespace: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Migrate or rename a namespace
    ///
    /// Moves all documents from one namespace to another. Useful for renaming
    /// namespaces or consolidating data.
    ///
    /// Examples:
    ///   rmcp-memex migrate-namespace --from old-name --to new-name
    ///   rmcp-memex migrate-namespace --from old --to new --merge
    ///   rmcp-memex migrate-namespace --from old --to new --dry-run
    ///   rmcp-memex migrate-namespace --from old --to new --delete-source false
    #[command(alias = "mv-namespace")]
    MigrateNamespace {
        /// Source namespace name
        #[arg(long, required = true)]
        from: String,

        /// Target namespace name
        #[arg(long, required = true)]
        to: String,

        /// If target namespace exists, merge documents instead of erroring
        #[arg(long)]
        merge: bool,

        /// Delete source namespace after migration (default: true)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        delete_source: bool,

        /// Show what would happen without making changes
        #[arg(long)]
        dry_run: bool,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Delete all documents in a namespace (DESTRUCTIVE)
    ///
    /// Permanently removes all chunks from the specified namespace.
    /// This action cannot be undone - use with caution!
    ///
    /// Examples:
    ///   rmcp-memex purge-namespace -n garbage
    ///   rmcp-memex purge-namespace -n old-data --confirm
    #[command(alias = "purge")]
    PurgeNamespace {
        /// Namespace to purge
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Skip confirmation prompt (use with caution!)
        #[arg(long)]
        confirm: bool,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Import documents from JSONL file into a namespace
    ///
    /// Reads documents exported with 'export' command and stores them.
    /// Can re-embed text if embeddings were not included in export.
    ///
    /// Examples:
    ///   rmcp-memex import -n memories -i backup.jsonl
    ///   rmcp-memex import -n new-namespace -i backup.jsonl --skip-existing
    Import {
        /// Target namespace (can differ from original export)
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Input JSONL file path
        #[arg(long, short = 'i', required = true)]
        input: PathBuf,

        /// Skip documents whose content_hash already exists in target namespace
        #[arg(long)]
        skip_existing: bool,

        /// Database path override
        #[arg(long)]
        db_path: Option<String>,
    },

    /// Reprocess exported JSONL into a fresh namespace using the current chunker
    ///
    /// Useful when the original source files are gone but the namespace export is valuable.
    /// The command collapses onion families back to a single canonical document, optionally
    /// preprocesses the text, and re-indexes it with the requested slice mode.
    ///
    /// Examples:
    ///   rmcp-memex export -n kodowanie -o kodowanie.jsonl
    ///   rmcp-memex reprocess -i kodowanie.jsonl -n kodowanie-v2 --slice-mode onion-fast
    ///   rmcp-memex reprocess -i memories.jsonl -n memories-v2 --preprocess --dry-run
    #[command(alias = "reindex-export")]
    Reprocess {
        /// Target namespace for rebuilt documents
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Input JSONL file produced by 'export'
        #[arg(long, short = 'i', required = true)]
        input: PathBuf,

        /// Slice mode for the rebuilt namespace
        #[arg(long, short = 's', default_value = "onion", value_parser = ["onion", "onion-fast", "fast", "flat"])]
        slice_mode: String,

        /// Apply preprocessing before rebuilding documents
        #[arg(long)]
        preprocess: bool,

        /// Skip documents already rebuilt with the same source hash
        #[arg(long)]
        skip_existing: bool,

        /// Show what would be rebuilt without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Database path override
        #[arg(long)]
        db_path: Option<String>,
    },

    /// Reindex an existing rmcp-memex namespace into '<namespace>-reindexed'
    ///
    /// This is the in-database equivalent of 'export -> reprocess' for namespaced
    /// rmcp-memex stores. It reads the existing namespace, collapses onion families
    /// back to canonical documents, and writes a rebuilt namespace without touching
    /// the source data.
    ///
    /// Examples:
    ///   rmcp-memex reindex -n kodowanie
    ///   rmcp-memex reindex -n kodowanie --dry-run
    ///   rmcp-memex reindex -n kodowanie --target-namespace kodowanie-v2 --slice-mode onion-fast
    Reindex {
        /// Source namespace to rebuild
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Target namespace override (default: '<namespace>-reindexed')
        #[arg(long)]
        target_namespace: Option<String>,

        /// Slice mode for the rebuilt namespace
        #[arg(long, short = 's', default_value = "onion", value_parser = ["onion", "onion-fast", "fast", "flat"])]
        slice_mode: String,

        /// Apply preprocessing before rebuilding documents
        #[arg(long)]
        preprocess: bool,

        /// Skip documents already rebuilt with the same source hash
        #[arg(long)]
        skip_existing: bool,

        /// Show what would be rebuilt without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Database path override
        #[arg(long)]
        db_path: Option<String>,
    },

    /// Audit database quality and text integrity
    ///
    /// Analyzes namespaces for embedding quality, text integrity (>90% target),
    /// and provides recommendations for cleanup.
    ///
    /// Examples:
    ///   rmcp-memex audit                    # Audit all namespaces
    ///   rmcp-memex audit -n memories        # Audit specific namespace
    ///   rmcp-memex audit --threshold 85     # Custom quality threshold
    ///   rmcp-memex audit --json             # JSON output for scripting
    #[command(alias = "quality")]
    Audit {
        /// Specific namespace to audit (default: all namespaces)
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Minimum quality threshold (0-100, default: 90)
        #[arg(long, default_value = "90")]
        threshold: u8,

        /// Show detailed metrics for each chunk (verbose)
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Purge low-quality namespaces based on audit results
    ///
    /// Removes namespaces that fall below the quality threshold.
    /// Always runs in dry-run mode unless --confirm is passed.
    ///
    /// Examples:
    ///   rmcp-memex purge-quality                      # Dry run with 90% threshold
    ///   rmcp-memex purge-quality --threshold 80      # Lower threshold
    ///   rmcp-memex purge-quality --confirm           # Actually delete
    #[command(alias = "purge-low-quality")]
    PurgeQuality {
        /// Minimum quality threshold (0-100, default: 90)
        #[arg(long, default_value = "90")]
        threshold: u8,

        /// Actually delete namespaces (default: dry-run)
        #[arg(long)]
        confirm: bool,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Manage auth tokens with per-token scopes and namespace ACL
    ///
    /// Create, list, revoke, and rotate bearer tokens for HTTP API access.
    /// Each token is hashed with argon2id at rest. The plaintext is shown
    /// ONCE on creation and can never be retrieved again.
    ///
    /// Examples:
    ///   rmcp-memex auth create --description "iPhone" --scopes read,write --namespaces kb:claude,kb:mikserka
    ///   rmcp-memex auth list
    ///   rmcp-memex auth revoke --id monika-iphone
    ///   rmcp-memex auth rotate --id monika-iphone
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },
}

/// Auth token management subcommands.
#[derive(Subcommand, Debug)]
pub enum AuthAction {
    /// Create a new auth token
    ///
    /// Generates a new token with specified scopes and namespace access.
    /// The plaintext token is printed ONCE and never stored.
    Create {
        /// Human-readable token identifier (e.g., "monika-iphone")
        #[arg(long)]
        id: Option<String>,

        /// Description of what this token is for
        #[arg(long, required = true)]
        description: String,

        /// Comma-separated scopes: read, write, admin
        #[arg(long, default_value = "read,write")]
        scopes: String,

        /// Comma-separated namespace ACL. Use "*" for all namespaces.
        #[arg(long, default_value = "*")]
        namespaces: String,

        /// Token expiry (RFC 3339 timestamp, e.g., "2026-12-31T00:00:00Z")
        #[arg(long)]
        expires_at: Option<String>,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// List all tokens (without revealing plaintext)
    List {
        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Revoke (delete) a token by its ID
    Revoke {
        /// Token ID to revoke
        #[arg(long, required = true)]
        id: String,
    },

    /// Rotate a token: revoke old, create new with same metadata
    Rotate {
        /// Token ID to rotate
        #[arg(long, required = true)]
        id: String,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },
}

impl Cli {
    pub fn into_server_config(self) -> Result<ServerConfig> {
        let (file_cfg, config_path) = load_or_discover_config(self.config.as_deref())?;
        if let Some(ref path) = config_path {
            eprintln!("Using config: {}", path);
        }

        let legacy_mode = self.mode.clone().or_else(|| file_cfg.mode.clone());
        let legacy_features = self.features.clone().or_else(|| file_cfg.features.clone());
        if legacy_mode.is_some() || legacy_features.is_some() {
            eprintln!(
                "Warning: legacy mode/features settings are ignored. rmcp-memex now exposes one canonical MCP surface; constrain access with --allowed-paths, HTTP auth, or namespace security instead."
            );
        }

        // Extract embedding config first (before any moves from file_cfg)
        let embeddings = file_cfg.resolve_embedding_config();
        let default_cfg = ServerConfig::default();

        // Build security config from CLI and file settings
        let security_enabled = self.security_enabled || file_cfg.security_enabled.unwrap_or(false);
        let token_store_path = self.token_store_path.or(file_cfg.token_store_path);

        Ok(ServerConfig {
            cache_mb: self
                .cache_mb
                .or(file_cfg.cache_mb)
                .unwrap_or(default_cfg.cache_mb),
            db_path: self
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or(default_cfg.db_path),
            max_request_bytes: self
                .max_request_bytes
                .or(file_cfg.max_request_bytes)
                .unwrap_or(default_cfg.max_request_bytes),
            log_level: self
                .log_level
                .or(file_cfg.log_level)
                .map(|s| parse_log_level(&s))
                .unwrap_or(default_cfg.log_level),
            allowed_paths: self
                .allowed_paths
                .or(file_cfg.allowed_paths)
                .unwrap_or(default_cfg.allowed_paths),
            security: NamespaceSecurityConfig {
                enabled: security_enabled,
                token_store_path,
            },
            embeddings,
            hybrid: default_cfg.hybrid,
        })
    }
}

pub fn parse_log_level(level: &str) -> Level {
    match level.to_ascii_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    }
}

/// Check if a path matches a glob pattern
pub fn matches_glob(path: &Path, pattern: &str) -> bool {
    let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };
    glob::Pattern::new(pattern)
        .map(|p| p.matches(file_name))
        .unwrap_or(false)
}

/// Collect files to index based on path, recursion, and glob settings
pub fn collect_files(
    path: &Path,
    recursive: bool,
    glob_pattern: Option<&str>,
    max_depth: usize,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        // Single file - check glob if provided
        if let Some(pattern) = glob_pattern {
            if matches_glob(path, pattern) {
                files.push(path.to_path_buf());
            }
        } else {
            files.push(path.to_path_buf());
        }
        return Ok(files);
    }

    // Directory walk
    let mut walker = WalkDir::new(path);
    if !recursive {
        walker = walker.max_depth(1);
    } else if max_depth > 0 {
        walker = walker.max_depth(max_depth);
    }

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let entry_path = entry.path();
        if !entry_path.is_file() {
            continue;
        }

        // Filter by glob pattern if provided
        if glob_pattern.is_some_and(|pattern| !matches_glob(entry_path, pattern)) {
            continue;
        }

        files.push(entry_path.to_path_buf());
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn legacy_mode_and_features_flags_parse_but_do_not_change_server_shape() {
        let tmp = tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(
            &config_path,
            "mode = \"memory\"\nfeatures = \"memory,search\"\n",
        )
        .unwrap();

        let cli = Cli::parse_from([
            "rmcp-memex",
            "--config",
            config_path.to_str().unwrap(),
            "--mode",
            "full",
            "--features",
            "filesystem,memory,search",
            "serve",
        ]);
        let config = cli.into_server_config().unwrap();
        let defaults = ServerConfig::default();

        assert_eq!(config.db_path, defaults.db_path);
        assert_eq!(config.cache_mb, defaults.cache_mb);
        assert_eq!(config.max_request_bytes, defaults.max_request_bytes);
        assert_eq!(config.allowed_paths, defaults.allowed_paths);
    }
    #[test]
    fn dashboard_command_parses_without_explicit_port() {
        let cli = Cli::parse_from(["rmcp-memex", "dashboard"]);

        match cli.command {
            Some(Commands::Dashboard { port, no_open }) => {
                assert_eq!(port, None);
                assert!(!no_open);
            }
            other => panic!("expected dashboard command, got {:?}", other),
        }
    }

    #[test]
    fn sse_command_parses_without_explicit_port() {
        let cli = Cli::parse_from(["rmcp-memex", "sse"]);

        match cli.command {
            Some(Commands::Sse { port }) => assert_eq!(port, None),
            other => panic!("expected sse command, got {:?}", other),
        }
    }

    #[test]
    fn repair_writes_command_parses() {
        let cli = Cli::parse_from(["rmcp-memex", "repair-writes", "--execute", "-n", "memories"]);

        match cli.command {
            Some(Commands::RepairWrites {
                namespace,
                execute,
                json,
            }) => {
                assert_eq!(namespace.as_deref(), Some("memories"));
                assert!(execute);
                assert!(!json);
            }
            other => panic!("expected repair-writes command, got {:?}", other),
        }
    }

    #[test]
    fn auth_mode_flag_parses_all_routes() {
        let cli = Cli::parse_from(["rmcp-memex", "--auth-mode", "all-routes", "serve"]);
        assert_eq!(cli.auth_mode, "all-routes");
    }

    #[test]
    fn auth_mode_defaults_to_mutating_only() {
        let cli = Cli::parse_from(["rmcp-memex", "serve"]);
        assert_eq!(cli.auth_mode, "mutating-only");
    }

    #[test]
    fn allow_network_without_auth_parses() {
        let cli = Cli::parse_from(["rmcp-memex", "--allow-network-without-auth", "serve"]);
        assert!(cli.allow_network_without_auth);
    }

    #[test]
    fn allow_query_token_parses() {
        let cli = Cli::parse_from(["rmcp-memex", "--allow-query-token", "serve"]);
        assert!(cli.allow_query_token);
    }

    #[test]
    fn auth_mode_rejects_invalid_value() {
        let result = Cli::try_parse_from(["rmcp-memex", "--auth-mode", "bogus", "serve"]);
        assert!(result.is_err());
    }
}
