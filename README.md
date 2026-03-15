# rmcp-memex
[![Crates.io](https://img.shields.io/crates/v/rmcp-memex)](https://crates.io/crates/rmcp-memex) [![License](https://img.shields.io/crates/l/rmcp-memex)](LICENSE) [![Downloads](https://img.shields.io/crates/d/rmcp-memex)](https://crates.io/crates/rmcp-memex) [![CI](https://github.com/VetCoders/rmcp-memex/actions/workflows/ci.yml/badge.svg)](https://github.com/VetCoders/rmcp-memex/actions)

RAG/Memory MCP Server with LanceDB vector storage for AI agents.

## Overview

`rmcp-memex` is an MCP (Model Context Protocol) server providing:
- **RAG (Retrieval-Augmented Generation)** - document indexing and semantic search
- **Hybrid Search** - BM25 keyword + semantic vector search (Tantivy-based)
- **Vector Memory** - semantic storage and retrieval of text chunks
- **Namespace Isolation** - data isolation in namespaces
- **Security** - token-based access control for protected namespaces
- **Onion Slice Architecture** - hierarchical embeddings (OUTER→MIDDLE→INNER→CORE)
- **Preprocessing** - automatic noise filtering from conversation exports (~36-40% reduction)
- **Exact-Match Deduplication** - SHA256-based dedup for overlapping exports

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      rmcp-memex                              │
├─────────────────────────────────────────────────────────────┤
│  MCP Server (JSON-RPC over stdio)                           │
│  ├── handlers/mod.rs    - Request routing & validation      │
│  ├── security/mod.rs    - Namespace access control          │
│  └── rag/mod.rs         - RAG pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                               │
│  ├── LanceDB           - Vector embeddings                  │
│  ├── Tantivy           - BM25 keyword index                 │
│  └── moka              - In-memory cache                    │
├─────────────────────────────────────────────────────────────┤
│  Embeddings (External Providers)                             │
│  ├── Ollama            - Local models (recommended)         │
│  ├── MLX Bridge        - Apple Silicon acceleration         │
│  └── OpenAI-compatible - Any compatible endpoint            │
└─────────────────────────────────────────────────────────────┘
```

## Features

### RAG Tools
| Tool | Description |
|------|-------------|
| `rag_index` | Index document from file |
| `rag_index_text` | Index raw text |
| `rag_search` | Search documents semantically (supports `auto_route`) |

### Memory Tools
| Tool | Description |
|------|-------------|
| `memory_upsert` | Add/update chunk in namespace |
| `memory_get` | Get chunk by ID |
| `memory_search` | Search semantically in namespace (supports `auto_route`) |
| `memory_delete` | Delete chunk |
| `memory_purge_namespace` | Delete all chunks in namespace |
| `dive` | Deep exploration with all onion layers (outer/middle/inner/core) |

### Security Tools
| Tool | Description |
|------|-------------|
| `namespace_create_token` | Create access token for namespace |
| `namespace_revoke_token` | Revoke token (namespace becomes public) |
| `namespace_list_protected` | List protected namespaces |
| `namespace_security_status` | Security system status |

## Library Usage

`rmcp-memex` can be used as a library in your Rust applications. It provides a high-level `MemexEngine` API for vector storage operations.

### Add to Cargo.toml

```toml
# Full library with CLI
rmcp-memex = "0.4"

# Library only (no CLI dependencies)
rmcp-memex = { version = "0.4", default-features = false }
```

### Basic Usage

```rust
use rmcp_memex::{MemexEngine, MemexConfig, MetaFilter, StoreItem};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Quick setup for any application
    let engine = MemexEngine::for_app("my-app", "documents").await?;

    // Store a document
    engine.store(
        "doc-1",
        "Patient presented with lethargy and decreased appetite",
        json!({"patient_id": "P-123", "visit_type": "checkup"})
    ).await?;

    // Search semantically
    let results = engine.search("lethargy symptoms", 10).await?;
    for r in &results {
        println!("{}: {} (score: {:.2})", r.id, r.text, r.score);
    }

    // Get by ID
    if let Some(doc) = engine.get("doc-1").await? {
        println!("Found: {}", doc.text);
    }

    // Delete
    engine.delete("doc-1").await?;

    Ok(())
}
```

### Vista Integration

For Vista PIMS, use the optimized constructor:

```rust
use rmcp_memex::MemexEngine;

// Vista-optimized: 1024 dims, qwen3-embedding:0.6b model
let engine = MemexEngine::for_vista().await?;

// Store visit notes
engine.store(
    "visit-456",
    "SOAP note: Feline diabetes mellitus diagnosis...",
    json!({"patient_id": "P-789", "doc_type": "soap_note"})
).await?;
```

### Batch Operations

```rust
use rmcp_memex::{MemexEngine, StoreItem};
use serde_json::json;

let engine = MemexEngine::for_app("my-app", "notes").await?;

let items = vec![
    StoreItem::new("doc-1", "First document").with_metadata(json!({"type": "note"})),
    StoreItem::new("doc-2", "Second document").with_metadata(json!({"type": "note"})),
    StoreItem::new("doc-3", "Third document").with_metadata(json!({"type": "note"})),
];

let result = engine.store_batch(items).await?;
println!("Stored {} documents", result.success_count);
```

### GDPR-Compliant Deletion

```rust
use rmcp_memex::{MemexEngine, MetaFilter};

let engine = MemexEngine::for_app("my-app", "patients").await?;

// Delete all documents for a specific patient
let filter = MetaFilter::for_patient("P-123");
let deleted = engine.delete_by_filter(filter).await?;
println!("Deleted {} documents", deleted);
```

### Hybrid Search (BM25 + Vector)

```rust
use rmcp_memex::{MemexEngine, SearchMode};

let engine = MemexEngine::for_app("my-app", "documents").await?;

// Hybrid search with BM25 + vector fusion (recommended)
let results = engine.search_hybrid("dragon mac studio", 10).await?;
for r in &results {
    println!("{}: {} (combined: {:.2}, vector: {:.2}, bm25: {:.2})",
        r.id, r.document, r.combined_score, r.vector_score, r.bm25_score);
}

// Explicit mode selection
let results = engine.search_with_mode("exact keyword", 10, SearchMode::Keyword).await?;
let results = engine.search_with_mode("semantic concept", 10, SearchMode::Vector).await?;
let results = engine.search_with_mode("best of both", 10, SearchMode::Hybrid).await?;
```

### Agent Tools API

For MCP-compatible AI agents:

```rust
use rmcp_memex::{MemexEngine, tool_definitions, memory_store, memory_search};
use serde_json::json;

let engine = MemexEngine::for_app("agent", "memory").await?;

// Get tool definitions for MCP registration
let tools = tool_definitions();
for tool in &tools {
    println!("Tool: {} - {}", tool.name, tool.description);
}

// Use tool functions
let result = memory_store(
    &engine,
    "mem-1".to_string(),
    "Important information to remember".to_string(),
    json!({"source": "conversation"}),
).await?;
assert!(result.success);

let results = memory_search(&engine, "important".to_string(), 5, None).await?;
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `cli` | CLI binary, TUI wizard, progress bars | Yes |
| `provider-cascade` | Ollama/OpenAI-compatible embeddings | Yes |

```bash
# Build library only (no CLI)
cargo build --no-default-features

# Build with CLI
cargo build --features cli
```

---

## Configuration Guide

Complete guide for integrating rmcp-memex as a library in any Rust project.

### Prerequisites

**Ollama** (recommended) or any OpenAI-compatible embedding API:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull an embedding model (choose based on your needs)
ollama pull qwen3-embedding:0.6b    # 1024 dims, ~600MB (fast, good quality)
ollama pull qwen3-embedding:8b      # 4096 dims, ~4GB (best quality)
ollama pull nomic-embed-text        # 768 dims, ~274MB (lightweight)

# Verify it's running
curl http://localhost:11434/api/tags
```

### Environment Variables

Configure via `.env` or environment:

```bash
# =============================================================================
# EMBEDDING PROVIDER CONFIGURATION
# =============================================================================

# Ollama (default, recommended)
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSION=1024

# Database storage (auto-created)
MEMEX_DB_PATH=~/.rmcp-servers/myapp/lancedb

# Optional: BM25 keyword search index
MEMEX_BM25_PATH=~/.rmcp-servers/myapp/bm25

# =============================================================================
# ADVANCED: Multiple providers (fallback cascade)
# =============================================================================

# Remote embedding server fallback
# DRAGON_BASE_URL=http://your-server.local
# DRAGON_EMBEDDER_PORT=12345

# MLX embedder for Apple Silicon
# EMBEDDER_PORT=12300
# MLX_MAX_BATCH_CHARS=32000
# MLX_MAX_BATCH_ITEMS=16
# DISABLE_MLX=1  # Set to disable MLX fallback
```

### Quick Start (Auto-config)

```rust
use rmcp_memex::MemexEngine;
use serde_json::json;

// Auto-configures from defaults + environment
let engine = MemexEngine::for_app("my-app", "default").await?;

engine.store("doc-1", "Document content...", json!({"type": "note"})).await?;
let results = engine.search("content", 10).await?;
```

### Custom Configuration

```rust
use rmcp_memex::{MemexConfig, MemexEngine};
use rmcp_memex::embeddings::{EmbeddingConfig, ProviderConfig};

// Read from your app's environment
let ollama_url = std::env::var("OLLAMA_BASE_URL")
    .unwrap_or_else(|_| "http://localhost:11434".to_string());
let model = std::env::var("EMBEDDING_MODEL")
    .unwrap_or_else(|_| "qwen3-embedding:0.6b".to_string());
let dimension: usize = std::env::var("EMBEDDING_DIMENSION")
    .unwrap_or_else(|_| "1024".to_string())
    .parse()
    .unwrap_or(1024);
let db_path = std::env::var("MEMEX_DB_PATH")
    .unwrap_or_else(|_| "~/.rmcp-servers/myapp/lancedb".to_string());

let config = MemexConfig {
    app_name: "my-app".to_string(),
    namespace: "default".to_string(),
    db_path: Some(db_path),
    dimension,
    embedding_config: EmbeddingConfig {
        required_dimension: dimension,
        providers: vec![ProviderConfig {
            name: "ollama".to_string(),
            base_url: ollama_url,
            model,
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        ..EmbeddingConfig::default()
    },
    enable_bm25: false,
    bm25_config: None,
};

let engine = MemexEngine::new(config).await?;
```

### Provider Cascade (Multiple Fallbacks)

Configure multiple providers - library tries them in priority order:

```rust
use rmcp_memex::embeddings::{EmbeddingConfig, ProviderConfig};

let config = EmbeddingConfig {
    required_dimension: 1024,
    providers: vec![
        // Priority 1: Local Ollama (fastest)
        ProviderConfig {
            name: "ollama-local".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "qwen3-embedding:0.6b".to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        },
        // Priority 2: Remote server fallback
        ProviderConfig {
            name: "remote-server".to_string(),
            base_url: "http://your-server:8080".to_string(),
            model: "text-embedding-3-small".to_string(),
            priority: 2,
            endpoint: "/v1/embeddings".to_string(),
        },
        // Priority 3: OpenAI API fallback
        ProviderConfig {
            name: "openai".to_string(),
            base_url: "https://api.openai.com".to_string(),
            model: "text-embedding-3-small".to_string(),
            priority: 3,
            endpoint: "/v1/embeddings".to_string(),
        },
    ],
    ..EmbeddingConfig::default()
};
```

### Namespace Strategy

**Recommended: One namespace per application, use metadata for filtering:**

```rust
// ✅ CORRECT: Single namespace, filter by user_id/entity_id in metadata
let engine = MemexEngine::for_app("my-app", "default").await?;

// Store with entity IDs in metadata
engine.store("doc-1", "Document content...", json!({
    "user_id": "U-123",      // For multi-tenant filtering
    "project_id": "P-456",   // For project-level filtering
    "doc_type": "note",
    "created_at": "2024-12-28"
})).await?;

// Search within user context
let filter = MetaFilter::default().with_custom("user_id", "U-123");
let results = engine.search_filtered("query", filter, 10).await?;

// GDPR deletion: remove all user data
let deleted = engine.delete_by_filter(
    MetaFilter::default().with_custom("user_id", "U-123")
).await?;
```

### Embedding Models Reference

| Model | Dimensions | Size | Use Case |
|-------|------------|------|----------|
| `qwen3-embedding:0.6b` | 1024 | ~600MB | Fast, good quality (recommended) |
| `qwen3-embedding:8b` | 4096 | ~4GB | Best quality, slower |
| `nomic-embed-text` | 768 | ~274MB | Lightweight, fast |
| `mxbai-embed-large` | 1024 | ~670MB | Good multilingual |
| `all-minilm` | 384 | ~46MB | Very fast, lower quality |

### Troubleshooting

**Error: "No embedding providers available"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull model if missing
ollama pull qwen3-embedding:0.6b
```

**Error: "Dimension mismatch"**
- LanceDB dimension is fixed per table after creation
- Use different `db_path` for different dimensions
- Delete old database to change dimensions

**Error: "Connection refused"**
```bash
# Linux
systemctl status ollama
systemctl start ollama

# macOS
brew services info ollama
brew services start ollama

# Or run manually
ollama serve
```

**Performance tuning:**
```bash
# Larger batches (requires more VRAM)
MLX_MAX_BATCH_CHARS=64000
MLX_MAX_BATCH_ITEMS=32
```

---

## Quick Start

### Installation

**Quick install (recommended):**
```bash
curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
```

**From source:**
```bash
cargo install --path .
```

### Running

```bash
# Default mode (all features)
rmcp-memex serve

# Memory-only mode (no filesystem access)
rmcp-memex serve --mode memory

# With security enabled
rmcp-memex serve --security-enabled

# With HTTP/SSE server for multi-agent access
rmcp-memex serve --http-port 6660

# HTTP-only daemon mode (no MCP stdio)
rmcp-memex serve --http-port 6660 --http-only
```

## HTTP/SSE Server (Multi-Agent Access)

LanceDB uses exclusive file locks - only one process can access the database at a time.
The HTTP/SSE server solves this by providing a central access point for multiple agents.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     rmcp-memex daemon                        │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │   MCP Server    │    │   HTTP/SSE      │                 │
│  │   (stdio)       │    │   (port 6660)   │                 │
│  └────────┬────────┘    └────────┬────────┘                 │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│              ┌─────────────┐                                │
│              │ RAGPipeline │ ← Single lock holder           │
│              └──────┬──────┘                                │
│                     ▼                                       │
│              ┌─────────────┐                                │
│              │   LanceDB   │                                │
│              └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │
    Claude Desktop       HTTP Agents
    (MCP stdio)          (curl, fetch)
```

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (status, db_path, embedding_provider) |
| `/search` | POST | Vector search with optional layer filter |
| `/sse/search` | GET | SSE streaming search (real-time results) |
| `/upsert` | POST | Add/update document |
| `/index` | POST | Full pipeline indexing with onion slices |
| `/expand/{ns}/{id}` | GET | Expand onion slice (get children) |
| `/parent/{ns}/{id}` | GET | Drill up to parent slice |
| `/get/{ns}/{id}` | GET | Get document by ID |
| `/delete/{ns}/{id}` | POST | Delete document |
| `/ns/{namespace}` | DELETE | Purge entire namespace |

### MCP-over-SSE Endpoints (Claude Code compatibility)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sse/` | GET | SSE stream - sends `endpoint` event with messages URL |
| `/messages/` | POST | JSON-RPC messages with `?session_id=xxx` |

Configure in `~/.claude.json`:
```json
{
  "mcpServers": {
    "rmcp-memex": {
      "type": "sse",
      "url": "http://localhost:6660/sse/"
    }
  }
}
```

### Usage Examples

```bash
# Start daemon
rmcp-memex serve --http-port 6660 --http-only --db-path ~/.ai-memories/lancedb &

# Health check
curl http://localhost:6660/health

# Store document
curl -X POST http://localhost:6660/upsert \
  -H "Content-Type: application/json" \
  -d '{"namespace": "agent1", "id": "mem1", "content": "Important context..."}'

# Search
curl -X POST http://localhost:6660/search \
  -H "Content-Type: application/json" \
  -d '{"query": "context", "namespace": "agent1", "limit": 10}'

# SSE streaming search
curl -N "http://localhost:6660/sse/search?query=context&namespace=agent1&limit=5"
```

### Multi-Host Database Paths

For setups with multiple machines (e.g., dragon, mgbook16), use per-host database paths:

```bash
# Per-host paths (each machine gets own database)
rmcp-memex serve --db-path ~/.ai-memories/lancedb.$(hostname -s)

# Or use the wizard for machine-agnostic configuration
rmcp-memex wizard
```

The TUI wizard auto-detects hostname and offers:
- **Shared mode**: `~/.ai-memories/lancedb` (same path everywhere)
- **Per-host mode**: `~/.ai-memories/lancedb.dragon`, `~/.ai-memories/lancedb.mgbook16`, etc.

### Configuration (TOML)

```toml
# ~/.rmcp-servers/config/rmcp-memex.toml

mode = "full"
db_path = "~/.rmcp-servers/rmcp-memex/lancedb"
cache_mb = 4096
log_level = "info"

# Whitelist of allowed paths
allowed_paths = [
    "~",
    "/Volumes/ExternalDrive/data"
]

# Security
security_enabled = true
token_store_path = "~/.rmcp-servers/rmcp-memex/tokens.json"
```

## Documentation

- [01_security.md](./01_security.md) - Security system (namespace tokens)
- [02_configuration.md](./02_configuration.md) - Configuration and CLI options

## Onion Slice Architecture

Instead of traditional flat chunking, rmcp-memex offers hierarchical "onion slices":

```
┌─────────────────────────────────────────┐
│  OUTER (~100 chars)                     │  ← Minimum context, maximum navigation
│  Keywords + ultra-compression           │
├─────────────────────────────────────────┤
│  MIDDLE (~300 chars)                    │  ← Key sentences + context
├─────────────────────────────────────────┤
│  INNER (~600 chars)                     │  ← Expanded content
├─────────────────────────────────────────┤
│  CORE (full text)                       │  ← Complete document
└─────────────────────────────────────────┘
```

**Philosophy:** "Minimum info → Maximum navigation paths"

### QueryRouter & Auto-Route

Intelligent query intent detection for automatic search mode selection:

```bash
# Auto-detect query intent and select optimal mode
rmcp-memex search -n memories -q "when did we buy dragon" --auto-route
# Output: Query intent: temporal (confidence: 0.70)
#         Selects: hybrid mode with date boosting

# Structural queries suggest loctree
rmcp-memex search -n code -q "who imports main.rs" --auto-route
# Output: Query intent: structural (confidence: 0.80)
#         Consider: loctree query --kind who-imports --target main.rs

# Deep exploration with all onion layers
rmcp-memex dive -n memories -q "dragon" --verbose
```

**Intent Types:**
| Intent | Trigger Keywords | Recommended Mode |
|--------|-----------------|------------------|
| Temporal | when, date, yesterday, ago, 2024 | Hybrid (date boost) |
| Structural | import, depends, module, who uses | BM25 + loctree suggestion |
| Semantic | similar, related, explain | Vector |
| Exact | "quoted strings" | BM25 |
| Hybrid | (default) | Vector + BM25 fusion |

### CLI Commands

```bash
# Index with onion slicing (default)
rmcp-memex index -n memories /path/to/data/ --slice-mode onion

# Index with progress bar and ETA
rmcp-memex index -n memories /path/to/data/ --progress

# Index with flat chunking (backward compatible)
rmcp-memex index -n memories /path/to/data/ --slice-mode flat

# Search in namespace
rmcp-memex search -n memories -q "best moments" --limit 10

# Search only in specific layer
rmcp-memex search -n memories -q "query" --layer outer

# Drill down in hierarchy (expand children)
rmcp-memex expand -n memories -i "slice_id_here"

# Get chunk by ID
rmcp-memex get -n memories -i "chunk_abc123"

# RAG search (cross-namespace)
rmcp-memex rag-search -q "search term" --limit 5

# List namespaces with stats
rmcp-memex namespaces --stats

# Export namespace to JSON
rmcp-memex export -n memories -o backup.json --include-embeddings
```

### Preprocessing (Noise Filtering)

Automatic removal of ~36-40% noise from conversation exports:
- MCP tool artifacts (`<function_calls>`, `<invoke>`, etc.)
- CLI output (git status, cargo build, npm install)
- Metadata (UUIDs, timestamps → placeholders)
- Empty/boilerplate content

```bash
# Index with preprocessing
rmcp-memex index -n memories /path/to/export.json --preprocess
```

### Exact-Match Deduplication

SHA256-based dedup for overlapping exports (e.g., quarterly exports containing 6 months of data):

```bash
# Dedup enabled (default)
rmcp-memex index -n memories /path/to/data/

# Disable dedup
rmcp-memex index -n memories /path/to/data/ --no-dedup
```

**Output with statistics:**
```
Indexing complete:
  New chunks:          234
  Files indexed:       67
  Skipped (duplicate): 33
  Deduplication:       enabled
```

## Code Structure

```
rmcp-memex/
├── src/
│   ├── lib.rs              # Public API & ServerConfig
│   ├── bin/
│   │   └── rmcp-memex.rs   # CLI binary (serve, index, search, get, expand, etc.)
│   ├── handlers/
│   │   └── mod.rs          # MCP request handlers
│   ├── security/
│   │   └── mod.rs          # Namespace access control
│   ├── rag/
│   │   └── mod.rs          # RAG pipeline + OnionSlice architecture
│   ├── preprocessing/
│   │   └── mod.rs          # Noise filtering for conversation exports
│   ├── storage/
│   │   └── mod.rs          # LanceDB + Tantivy (schema v3 with content_hash)
│   ├── embeddings/
│   │   └── mod.rs          # MLX/FastEmbed bridge
│   └── tui/
│       └── mod.rs          # Configuration wizard
└── Cargo.toml
```

## Claude/MCP Integration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "rmcp-memex": {
      "command": "rmcp-memex",
      "args": ["serve", "--security-enabled"]
    }
  }
}
```

---

Vibecrafted with AI Agents by VetCoders (c)2025 The LibraxisAI Team
Co-Authored-By: [Maciej](void@div0.space) & [Klaudiusz](the1st@whoai.am)
