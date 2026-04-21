# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `rmcp-memex reindex` subcommand for rebuilding namespace contents with paginated source reads, preflight embedding checks, and target-namespace safety guards
- Paginated `all_documents_page(offset, limit)` on `StorageManager` for bounded full-table scans
- Regression coverage for BM25 rollback so a failed keyword-index write does not leave ghost LanceDB rows behind
- Regression coverage for the SSE search `k` alias so transport-level top-k requests stay aligned with HTTP and MCP

### Changed
- `MemexEngine::delete_by_filter` now paginates the full namespace before deleting, preventing missed records in large datasets
- `run_export` now paginates with 5000-doc pages instead of a hard 100K limit, preventing silent truncation on large namespaces
- `collapse_export_records` now prefers longest text over highest layer rank when deduplicating, preventing loss of content from historical multi-slicer namespaces
- Invalid `--search-mode` values now return an error instead of silently falling back to the default mode
- Import with missing `content_hash` now computes the hash from text instead of using an empty string that would cause false-positive dedup matches
- Reprocess progress now reports total progress including skipped documents, and logs failed document IDs instead of aborting on the first embedding error

### Breaking
- Tool names renamed: `memory_search` → `search_documents`, `memory_upsert` → `upsert_document`, and other `memory_*` → `*_document` renames. See `src/handlers/mod.rs` for the full mapping
- `mcp_protocol::*` types moved under `mcp_core::*`
- `ServerConfig::features` field removed
- `maintenance::new` signature changed to `new(namespace, db_path)`

## [0.5.0] - 2026-04-10

### Added
- Regression coverage for BM25 rollback so a failed keyword-index write does not leave ghost LanceDB rows behind
- Regression coverage for the SSE search `k` alias so transport-level top-k requests stay aligned with HTTP and MCP

## [0.5.0] - 2026-04-10

### Added
- Release-facing landing page source under `docs/` with a GitHub Pages workflow and social preview asset
- Release runbook documenting preflight checks, tagging flow, smoke tests, and post-publish verification
- SHA256 checksum generation for GitHub release artifacts and installer-side checksum verification
- Host-local `scripts/build-release-bundle.sh` helper for producing the signed prebuilt tarballs shipped in GitHub Releases
- Local `scripts/release-local.sh` operator flow for build, macOS signing, and optional `gh release` upload

### Changed
- Resolved the shared MCP transport merge so `mcp_core` remains the single public dispatch surface
- Hardened CI with library-only and `cargo package` smoke checks in addition to the existing lint/test gates
- Aligned release automation, install scripts, and docs around the canonical `rmcp-memex` binary name and documented aliases
- Switched the direct `reqwest` client to Rustls so release binaries are more self-contained and do not pull `native-tls` from the top-level crate
- Clarified that GitHub Release bundles are the canonical install path, but they are built and signed locally before upload rather than compiled inside GitHub Actions

## [0.4.0] - 2026-03-09

### Changed
- Semver bump for the current breaking public API surface
- Added self-contained CLI E2E coverage for folder indexing, chunking, and RAG search
- Split CLI alias binaries into dedicated wrapper entrypoints to remove duplicate-target warnings structurally

## [0.3.6] - 2026-01-03

### Added
- **MCP-over-SSE Transport** - Native MCP protocol over HTTP/SSE for Claude Code compatibility
  - `GET /sse/` - SSE stream with `endpoint` event for session initialization
  - `POST /messages/?session_id=xxx` - JSON-RPC message endpoint
  - Full MCP protocol support: `initialize`, `tools/list`, `tools/call`
  - Session management with automatic cleanup
- **BM25 Read-Only Mode** - No lock contention for multi-agent read access
  - `BM25Config.read_only = true` - Opens index without IndexWriter (no lock)
  - HTTP-only mode (`--http-only`) now automatically uses read-only BM25
  - Multiple Claude instances can read simultaneously
  - Write operations return error in read-only mode (by design)

### Changed
- `--http-only` mode now sets `hybrid.bm25.read_only = true` automatically
- MCP-over-SSE endpoints available at both `/mcp/` and `/sse/` paths

## [0.3.5] - 2026-01-02

### Added
- Full CRUD and management commands to CLI

## [0.3.4] - 2025-12-31

### Added
- **Database Management Commands** - Full CRUD for namespaces and documents
  - `rmcp-memex merge --source ~/db1 --source ~/db2 --target ~/merged` - Merge multiple LanceDB databases with optional deduplication
  - `rmcp-memex dedup -n memories` - Find and remove duplicate documents by content_hash
  - `rmcp-memex migrate-namespace --from old --to new` - Move/rename namespaces with optional merge
  - `rmcp-memex import -n memories -i backup.jsonl` - Import documents from JSONL with optional re-embedding
  - `rmcp-memex export -n memories -o backup.jsonl` - Export namespace to portable JSONL format
  - `rmcp-memex gc --older-than 30d` - Garbage collection for orphans, empty namespaces, old docs
  - `rmcp-memex cross-search -q "query"` - Search across all namespaces at once
- **HTTP/SSE Server** - Multi-agent access without LanceDB lock conflicts
  - `--http-port 6660` flag to start HTTP server alongside MCP stdio
  - `--http-only` flag for daemon mode (HTTP only, no MCP stdio)
  - Endpoints: `/health`, `/search`, `/sse/search`, `/upsert`, `/index`, `/expand`, `/parent`, `/get`, `/delete`, `/ns`
  - SSE streaming for real-time search results with events: `start`, `result`, `done`
  - Concurrent multi-agent access through shared RAGPipeline
  - Built on axum with tower-http CORS support
- **TUI Wizard Enhancements** - Machine-agnostic configuration
  - Auto-detect hostname for per-host database paths
  - **Path Mode**: Shared (`~/.ai-memories/lancedb`) or Per-Host (`~/.ai-memories/lancedb.{hostname}`)
  - HTTP port configuration in wizard
  - Host info displayed in health check
  - Config TOML includes hostname and path mode comments
- **Multi-Host Database Paths** - Separate databases per machine
  - Pattern: `~/.ai-memories/lancedb.dragon`, `~/.ai-memories/lancedb.mgbook16`, etc.
  - Avoids conflicts when syncing config across machines
  - `MemexCfg::effective_db_path()` handles path resolution

### Changed
- Wizard settings now include 7 fields: db_path, path_mode, http_port, cache_mb, log_level, max_request_bytes, mode
- Host config snippets include `--http-port` when configured
- Health check shows hostname and path mode info

### Dependencies
- Added `axum` 0.8 with json feature
- Added `tokio-stream` 0.1
- Added `tower-http` 0.6 with cors feature
- Added `async-stream` 0.3

## [0.3.3] - 2025-12-30

### Fixed
- **TUI Wizard Runtime Panic** - "Cannot start a runtime from within a runtime"
  - Use `tokio::task::block_in_place()` when calling `block_on()` from async context
  - `Handle::try_current()` to get existing runtime handle instead of creating new one
  - Fixes crash when running `rmcp-memex wizard` from `#[tokio::main]` async context

### Added
- **LanceDB Maintenance Commands** (from 0.3.2-dev)
  - `rmcp-memex optimize` - Run all optimizations (compact + prune old versions)
  - `rmcp-memex compact` - Merge small fragment files into larger ones
  - `rmcp-memex cleanup --older-than-days N` - Remove old versions (default: 7 days)
  - `rmcp-memex stats` - Show database statistics (row count, version count)
  - Fixes "too many open files" errors from LanceDB fragment accumulation
  - Library API: `StorageManager::optimize()`, `compact()`, `cleanup()`, `stats()`
  - New type: `TableStats` exported from lib.rs

## [0.3.2] - 2025-12-29

### Added
- **Async Pipeline Mode** - Concurrent indexing with `--pipeline` flag
  - Runs file reading, chunking, embedding, and storage in parallel stages
  - Uses `tokio::sync::mpsc` channels with bounded buffers (100 items) for backpressure
  - Each stage runs in its own `tokio::spawn` for maximum concurrency
  - Ideal for large batch operations with significant I/O and GPU overlap
  - Example: `rmcp-memex index ~/documents -n docs --pipeline`
  - Library API: `run_pipeline()`, `PipelineConfig`, `PipelineResult`, `PipelineStats`
  - Note: `--progress` and `--resume` flags are not supported in pipeline mode
- **Parallel File Processing** - `--parallel N` flag for concurrent file indexing
  - Processes N files concurrently using `tokio::Semaphore` for rate limiting
  - Default: 4 parallel workers, configurable 1-16 via `-P N` or `--parallel N`
  - Uses atomic counters (`AtomicUsize`, `AtomicBool`) for thread-safe progress tracking
  - Preserves checkpoint/resume functionality with `Arc<Mutex<IndexCheckpoint>>`
  - One file failure doesn't stop others - graceful error handling
  - Example: `rmcp-memex index ~/documents -n docs --parallel 8`
  - Combines with all existing flags: `--dedup`, `--resume`, `--progress`, `--preprocess`
  - Note: Ignored when `--pipeline` is enabled (pipeline has its own concurrency model)

## [0.3.1] - 2025-12-29

### Added
- **QueryRouter** - Intelligent query intent detection and automatic search mode selection
  - `detect_intent()` - Fast heuristic-based intent detection (Temporal/Structural/Semantic/Exact/Hybrid)
  - `QueryRouter::route()` - Full routing with confidence scores and recommendations
  - Intent types: Temporal (date queries), Structural (code/import queries → loctree), Semantic, Exact (quoted), Hybrid
  - Polish language support for temporal keywords (kiedy, wczoraj, dzisiaj, etc.)
- **CLI `--auto-route` flag** - Automatic search mode selection for `search` command
  - Analyzes query intent and selects optimal mode (vector/bm25/hybrid)
  - Displays intent, confidence, and loctree suggestions when applicable
  - Example: `rmcp-memex search -n memories -q "when did we buy dragon" --auto-route`
- **MCP `auto_route` parameter** - Added to `rag_search` and `memory_search` tools
  - When `true`, QueryRouter overrides explicit `mode` parameter
  - Enables intelligent mode selection for AI agents
- **MCP `dive` tool** - Deep exploration with all onion layers
  - Searches all 4 layers (outer/middle/inner/core) simultaneously
  - Returns structured results per layer with scores and metadata
  - Parameters: `namespace`, `query`, `limit` (per layer), `verbose`
- **MemexEngine Hybrid Search** - Library-level hybrid search methods
  - `search_hybrid()` - BM25 + vector fusion search returning `HybridSearchResult`
  - `search_with_mode()` - Explicit mode selection (Vector/Keyword/Hybrid)
  - `search_bm25_fusion()` - Deprecated alias for backward compatibility
  - `HybridConfig` in `MemexConfig` for library consumers
- **CLI `--resume` flag** - Resumable indexing for interrupted operations
  - Saves checkpoint after each file to `.index-checkpoint-{namespace}.json`
  - On restart with `--resume`, skips already-indexed files
  - Checkpoint auto-deleted on successful completion
  - Failed files preserved for retry: `rmcp-memex index ... --resume`
- **OnionFast mode** - 2x faster indexing for large datasets
  - `--slice-mode onion-fast` or `--slice-mode fast`
  - Creates only outer+core layers (2 instead of 4)
  - Same search quality, half the embedding calls
  - Ideal for bulk indexing where speed matters

### Changed
- **Embedding batch size 4x increase** - Dramatically fewer API calls
  - `max_batch_items`: 16 → 64 (4x more chunks per request)
  - `max_batch_chars`: 32K → 128K (4x larger batches)
  - Better GPU utilization, significant speedup for large datasets
- **Embedding Retry with Backoff** - Robust embedding client with exponential retry
  - 10 retries with exponential backoff (1s, 2s, 4s... up to 30s max)
  - Survives embedder restarts, API timeouts, temporary failures
  - Detailed logging per retry attempt for debugging
- Reranker config updated to Qwen3-Reranker-8B on port 12346 (from 4B)
- MCP tool schemas now include `auto_route` parameter documentation
- **Dynamic ETA** - Progress bar now uses rolling EMA for speed measurement
  - Updates every 2 seconds instead of one-time calibration
  - Reflects actual GPU performance after warm-up (2-3x faster than cold start)
  - EMA smoothing (30% new / 70% old) prevents jitter

## [0.3.0] - 2025-12-28

### Added
- **MemexEngine** - High-level API for library consumers
  - `MemexEngine::for_app()` - quick setup for any application with auto-config
  - `MemexEngine::for_vista()` - Vista-optimized defaults (1024 dims, qwen3-embedding:0.6b)
  - CRUD operations: `store()`, `search()`, `get()`, `delete()`
  - Batch operations: `store_batch()` for efficient bulk inserts
  - Filtered operations: `search_filtered()`, `delete_by_filter()` for GDPR-compliant deletion
  - Hybrid search: `search_hybrid()` combining BM25 keyword + vector similarity
  - Builder pattern: `MemexConfig` with `with_dimension()`, `with_db_path()`, `with_bm25()`
- **Agent Tools API** - MCP-compatible tool functions for AI agents
  - `memory_store` - Store text with automatic embedding generation
  - `memory_search` - Semantic search with optional metadata filtering
  - `memory_get` - Retrieve document by ID
  - `memory_delete` - Delete document by ID
  - `memory_store_batch` - Efficient batch storage
  - `memory_delete_by_filter` - GDPR deletion by metadata filter
  - `tool_definitions()` - Returns all tool schemas for MCP registration
  - `ToolDefinition` struct matching MCP schema format
  - `ToolResult` struct for consistent tool responses
- **Feature Flags** - Library-only builds without CLI dependencies
  - `default = ["cli", "provider-cascade"]`
  - `cargo build --no-default-features` for minimal library build
  - CLI deps (clap, indicatif, ratatui, crossterm) now optional
- **MetaFilter** - Metadata filtering for searches and deletions
  - `for_patient()` - GDPR-compliant patient data filtering
  - `for_visit()` - Visit-specific filtering
  - `with_custom()` - Custom key-value filters
  - Date range support (`date_from`, `date_to`)
- **StoreItem** - Builder for batch storage items
- **BatchResult** - Result type for batch operations

### Changed
- CLI dependencies now optional (enabled by `cli` feature flag)
- Binary requires `cli` feature (`required-features = ["cli"]`)
- `progress` and `tui` modules conditional on `cli` feature
- Restructured lib.rs exports for lib-first usage

### Documentation
- **Configuration Guide** - Complete integration guide for any Rust project
  - Environment variables reference (`.env` configuration)
  - Quick start with `for_app()` auto-config
  - Custom `MemexConfig` with provider configuration
  - Provider cascade setup (multiple fallback providers)
  - Namespace strategy best practices
  - Embedding models reference table (dimensions, sizes, use cases)
  - Troubleshooting section for common errors
- **Library Usage** - Comprehensive examples
  - Basic CRUD operations
  - Batch operations with `store_batch()`
  - GDPR-compliant deletion with `MetaFilter`
  - Agent Tools API for MCP integration

## [0.2.4] - 2025-12-28

### Added
- **BM25 Keyword Search** - Hybrid search combining semantic + keyword (Tantivy-based).
  - Stemming and language support for improved recall
  - Token-aware validation utilities
- **Context-Aware Chunking** - Sentence-aware chunking in RAG pipeline.
  - Context prefix injection for enriched chunks
  - Improved boundary detection
- **Embedding Retry Logic** - Failed chunks retry individually with exponential backoff.

## [0.2.3] - 2025-12-27

### Added
- **Smart Progress Bar** - `--progress` flag for index command with ETA.
  - Phase 1: Pre-scan (file count, total size, estimated chunks)
  - Phase 2: Calibration (measure embedding speed on first file)
  - Phase 3: Progress bar with ETA based on calibration
- **Verbose Error Logging** - Detailed diagnostics for embedding operations.
  - Full error chain with URL, model, batch size
  - Per-chunk diagnostics with text preview
  - Response body dump on parse failures

### Changed
- Added `indicatif` crate for progress bar rendering.

## [0.2.2] - 2025-12-27

### Added
- **Path Sanitization** - Security hardening for TUI wizard (Semgrep findings).
  - `sanitize_existing_path()` / `sanitize_new_path()` functions
  - Paths validated against allowed directories (home, /tmp, /var/folders)
  - Traversal sequence detection
- **Release Workflow** - GitHub Actions for multi-platform binary releases.
- **Install Script** - `curl | sh` installer with platform detection.
  ```bash
  curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
  ```

## [0.2.1] - 2025-12-26

### Added
- **TUI Wizard** - Interactive configuration wizard with provider detection.
  - Auto-detect Ollama and MLX embedding providers
  - Health check system for embedder connectivity
  - Host detection for Codex, Cursor, Claude, JetBrains, VSCode
  - Config merge (preserves existing servers)
  - Dimension hints and extended host snippets (ClaudeCode, Junie)
- **Upsert Command** - Direct text chunk insertion via CLI.
- **Git Hooks** - Pre-commit (auto-fix) and pre-push (format, lint, test, build, Semgrep).

### Fixed
- **UTF-8 Boundary Panic** - Use `chars().count()` for accurate character counting in embedding batching.

### Removed
- **sled Dependency** - Eliminated dead code causing file corruption and crash loops.
  - Storage now uses only LanceDB + moka in-memory cache.

## [0.2.0] - 2025-12-26

### Breaking Changes
- **Binary renamed** from `rmcp_memex` to `rmcp-memex` (hyphenated).
- **Default paths changed** from `~/.rmcp_servers/rmcp_memex/` to `~/.rmcp-servers/rmcp-memex/`.
- **MCP server name** changed from `rmcp_memex` to `rmcp-memex`.

### Added
- **Namespace Security** - Token-based access control for protected namespaces.
  - `namespace_create_token` - Create access token for namespace
  - `namespace_revoke_token` - Revoke token (namespace becomes public)
  - `namespace_list_protected` - List protected namespaces
  - `namespace_security_status` - Security system status
- **Preprocessing Module** - Automatic noise filtering from conversation exports (~36-40% reduction).
  - Filters MCP tool artifacts (`<function_calls>`, `<invoke>`, etc.)
  - Removes CLI output (git status, cargo build, npm install)
  - Replaces metadata (UUIDs, timestamps) with placeholders
- **Onion Slice Architecture** - Hierarchical embeddings for better navigation.
  - OUTER (~100 chars) - Keywords + ultra-compression
  - MIDDLE (~300 chars) - Key sentences + context
  - INNER (~600 chars) - Expanded content
  - CORE (full text) - Complete document
- **Exact-Match Deduplication** - SHA256-based dedup for overlapping exports.
- **TUI Configuration Wizard** - Interactive setup for host configurations.
- **Comprehensive Documentation**:
  - `docs/01_security.md` - Namespace security guide
  - `docs/02_configuration.md` - CLI options and TOML config

### Changed
- Schema version bumped to v3 (added `content_hash` field).
- Embeddings configuration now uses `EmbeddingConfig` with multi-provider support.
- Updated to Rust Edition 2024 (stable).

### Fixed
- Consistent naming across package, binary, MCP server, and paths.

## [0.1.13] - 2025-12-25

### Added
- Initial standalone release synced from loctree-suite.
- LanceDB vector storage with 4096-dim embeddings.
- RAG tools: `rag_index`, `rag_index_text`, `rag_search`.
- Memory tools: `memory_upsert`, `memory_get`, `memory_search`, `memory_delete`, `memory_purge_namespace`.
- Multi-provider embeddings support (Ollama, MLX, OpenAI-compatible).

---

Vibecrafted with AI Agents by VetCoders (c)2025 The LibraxisAI Team
