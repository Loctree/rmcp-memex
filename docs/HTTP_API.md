# rmcp-memex HTTP API & Dashboard

Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

## Overview

The HTTP server provides REST API access to memex for agents and tools that cannot use MCP directly.
It also serves an embedded HTML dashboard for visual browsing of memories.

**Default port:** `6666`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 rmcp-memex HTTP Server                  │
│                    (port 6666)                          │
├─────────────────────────────────────────────────────────┤
│  LanceDB (vector storage)  │  RAM cache (namespaces)   │
│       [disk-based]         │    [background refresh]    │
└─────────────────────────────────────────────────────────┘
```

### Memory Management

- **LanceDB** stores vectors on disk (~25GB for large DBs)
- **Namespace cache** is loaded into RAM at startup (background task)
- **Search index** kept in RAM for fast queries
- HTTP API is **~100x faster** than CLI for repeated queries

## Starting the Server

```bash
# Via launchd (recommended - auto-restart)
launchctl load ~/Library/LaunchAgents/ai.libraxis.rmcp-memex.plist

# Direct (for debugging)
rmcp-memex serve --http-port 6666

# Using Makefile
make start    # Start via launchd
make stop     # Stop service
make restart  # Restart service
make status   # Check if running
make health   # Quick health check
```

## Dashboard

Open in browser: `http://localhost:6666/`

The dashboard provides:
- Visual namespace browser (left sidebar)
- Document list with text preview
- Search across all namespaces
- Expand/drill-up for onion slices

### If Dashboard Shows "Loading namespaces..."

This means the namespace cache is still being built (normal for large DBs) or the database needs optimization:

```bash
# Check status
curl http://localhost:6666/api/status

# If hint says "run optimize":
rmcp-memex optimize
# or
make optimize
```

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/api/status` | GET | Cache status and hints |
| `/api/overview` | GET | Database stats |

**GET /health**
```json
{"status":"ok","db_path":"/path/to/lancedb","embedding_provider":"ollama-dragon"}
```

**GET /api/status**
```json
{
  "cache_ready": true,
  "namespace_count": 5,
  "hint": "OK"
}
```
If `cache_ready` is false, run `rmcp-memex optimize`.

**GET /api/overview**
```json
{
  "namespace_count": 0,
  "total_documents": 27455,
  "db_path": "/Users/you/.ai-memories/lancedb",
  "embedding_provider": "ollama-dragon"
}
```

### Browse & Namespaces

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/namespaces` | GET | List all namespaces with counts |
| `/api/browse/{ns}` | GET | Browse documents in namespace |
| `/api/browse` | GET | Browse all documents |

**GET /api/namespaces**
```json
{
  "namespaces": [
    {"name": "kodowanie", "count": 15000},
    {"name": "memories", "count": 5000}
  ],
  "total": 2
}
```

**GET /api/browse/kodowanie?limit=50**
```json
{
  "documents": [
    {
      "id": "doc-123",
      "namespace": "kodowanie",
      "text": "Document content...",
      "layer": "L0_Atom",
      "can_expand": true,
      "can_drill_up": false
    }
  ]
}
```

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Search with namespace filter |
| `/cross-search` | GET | Search across all namespaces |
| `/sse/search` | GET | Streaming search (SSE) |
| `/sse/cross-search` | GET | Streaming cross-search |

**POST /search**
```bash
curl -X POST http://localhost:6666/search \
  -H "Content-Type: application/json" \
  -d '{"query": "rust async", "namespace": "kodowanie", "limit": 10}'
```

Response:
```json
{
  "results": [
    {
      "id": "doc-456",
      "namespace": "kodowanie",
      "text": "Matching content...",
      "score": 0.85,
      "layer": "L0_Atom"
    }
  ],
  "count": 1,
  "elapsed_ms": 42
}
```

**GET /cross-search**
```bash
curl "http://localhost:6666/cross-search?q=rust%20async&limit=5&total_limit=20"
```

### Document Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upsert` | POST | Insert/update document |
| `/index` | POST | Index text with full pipeline |
| `/expand/{ns}/{id}` | GET | Expand onion slice (children) |
| `/parent/{ns}/{id}` | GET | Get parent slice |
| `/ns/{namespace}` | DELETE | Purge entire namespace |

**POST /upsert**
```bash
curl -X POST http://localhost:6666/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "memories",
    "id": "memory-001",
    "content": "Today I learned about Rust async...",
    "metadata": {"type": "learning", "date": "2026-01-13"}
  }'
```

Response:
```json
{"status": "ok", "id": "memory-001", "namespace": "memories"}
```

**POST /index** (full pipeline with chunking)
```bash
curl -X POST http://localhost:6666/index \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "docs",
    "text": "Long document text to be chunked...",
    "slice_mode": "onion",
    "dedup": true
  }'
```

## Using with Claude Code Hooks

The HTTP API is designed for Claude Code hooks (fast, non-blocking):

```bash
# In your hook script:
MEMEX_URL="${MEMEX_URL:-http://localhost:6666}"

# Health check (1s timeout)
curl -s --max-time 1 "$MEMEX_URL/health" >/dev/null 2>&1 || exit 0

# Search (2s timeout)
curl -s --max-time 2 "$MEMEX_URL/cross-search?q=your+query&limit=3"

# Upsert (5s timeout for writes)
curl -s --max-time 5 -X POST "$MEMEX_URL/upsert" \
  -H "Content-Type: application/json" \
  -d '{"namespace":"ns","id":"id","content":"text"}'
```

## Performance Notes

| Operation | CLI | HTTP API |
|-----------|-----|----------|
| Search | ~5000ms | ~50ms |
| Upsert | ~3000ms | ~100ms |
| Health check | N/A | ~5ms |

The difference is because CLI loads the full LanceDB index on each invocation,
while HTTP server keeps it in RAM.

## Troubleshooting

### "Too many open files"

Database is fragmented. Run:
```bash
rmcp-memex optimize
# or
make optimize
```

### Dashboard shows blank/loading forever

1. Check server is running: `curl http://localhost:6666/health`
2. Check cache status: `curl http://localhost:6666/api/status`
3. If `cache_ready: false` persists, run `rmcp-memex optimize`

### Port conflict

Change port in config and plist:
```bash
# ~/.rmcp-servers/rmcp-memex/config.toml
http_port = 6666

# ~/Library/LaunchAgents/ai.libraxis.rmcp-memex.plist
<string>--http-port</string>
<string>6666</string>
```

Then restart: `make restart`
