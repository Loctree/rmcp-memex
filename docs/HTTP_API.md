# rmcp-memex HTTP API & Dashboard

Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

## Overview

`rmcp-memex` exposes the same HTTP surface in two ergonomic entrypoints:
`rmcp-memex dashboard` for local browsing on port `8987`, and `rmcp-memex sse`
for the agent-facing daemon on port `8997`. In both cases, the dashboard at `/`
and new HTTP clients should start with `GET /api/discovery`.

`/api/discovery` is the canonical read surface because it bundles:

- daemon readiness (`status`, `hint`)
- runtime identity (`version`, `db_path`, `embedding_provider`)
- current dataset summary (`total_documents`, `namespace_count`)
- namespace inventory (`namespaces[]` with `id`, `count`, `last_indexed_at`)

The older read endpoints `/api/status`, `/api/overview`, and `/api/namespaces`
still exist, but they are compatibility slices now, not the first stop for new
clients.

## Start The Server

```bash
# Standard dual transport mode (stdio + HTTP)
rmcp-memex serve --http-port 8997

# HTTP-only daemon mode
rmcp-memex sse

# Protect mutating routes
rmcp-memex serve --http-port 8997 --auth-token "$MEMEX_AUTH_TOKEN"

# Open the local dashboard in a browser
rmcp-memex dashboard
```

Default dashboard URL:

```text
http://localhost:8987/
```

## Canonical Discovery

### `GET /api/discovery`

Use this first. It is the single source of truth for dashboards, health probes,
and lightweight HTTP clients.

Example response:

```json
{
  "status": "ok",
  "hint": "OK",
  "version": "0.5.1",
  "db_path": "/Users/you/.rmcp-servers/rmcp-memex/lancedb",
  "embedding_provider": "ollama-local",
  "total_documents": 27455,
  "namespace_count": 2,
  "namespaces": [
    {
      "id": "kodowanie",
      "count": 15000,
      "last_indexed_at": "2026-03-15T10:11:12Z"
    },
    {
      "id": "memories",
      "count": 12455,
      "last_indexed_at": null
    }
  ]
}
```

### Discovery Status Semantics

- `status = "ok"` means the namespace cache is ready and `namespaces[]` is fully populated.
- `status = "loading"` means the daemon is up, but the background namespace cache is still warming.
- `hint` is the user-facing guidance string. When cache warmup drags, it tells clients to run `rmcp-memex optimize`.

## Read Surfaces

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Embedded dashboard |
| `/health` | GET | Minimal liveness check |
| `/api/discovery` | GET | Canonical readiness + namespace summary |
| `/api/status` | GET | Compatibility slice of cache readiness |
| `/api/overview` | GET | Compatibility slice of DB totals |
| `/api/namespaces` | GET | Compatibility slice of namespace counts |
| `/api/browse` | GET | Browse all documents |
| `/api/browse/{ns}` | GET | Browse one namespace |
| `/get/{ns}/{id}` | GET | Fetch one document |
| `/expand/{ns}/{id}` | GET | Expand an onion slice to children |
| `/parent/{ns}/{id}` | GET | Fetch the parent slice |
| `/search` | POST | Search with optional namespace, mode, project, layer, and deep filters (`k` alias supported) |
| `/cross-search` | GET | Search across namespaces with `mode=vector|bm25|hybrid` |
| `/sse/search` | GET | Streaming single-namespace search |
| `/sse/cross-search` | GET | Streaming cross-namespace search |
| `/sse/namespaces` | GET | Streaming namespace summary |

### `GET /health`

Minimal health shape:

```json
{
  "status": "ok",
  "db_path": "/path/to/lancedb",
  "embedding_provider": "ollama-local"
}
```

### Compatibility Slices

These remain useful for legacy clients, but new code should prefer `GET /api/discovery`.

- `GET /api/status`
- `GET /api/overview`
- `GET /api/namespaces`

## Search Examples

### `POST /search`

```bash
curl -X POST http://localhost:8997/search \
  -H "Content-Type: application/json" \
  -d '{"query":"rust async","namespace":"kodowanie","k":10,"mode":"hybrid","project":"Vista","deep":true}'
```

Representative response:

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

### `GET /cross-search`

```bash
curl "http://localhost:8997/cross-search?q=rust%20async&limit=5&total_limit=20&mode=hybrid"
```

## Mutating Surfaces

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upsert` | POST | Upsert one chunk directly |
| `/index` | POST | Index content through the chunking pipeline |
| `/delete/{ns}/{id}` | POST | Delete one document |
| `/ns/{namespace}` | DELETE | Purge a namespace |
| `/refresh` | POST | Refresh cached namespace inventory |
| `/sse/optimize` | POST | Stream optimize progress |

### Auth Model

Read-only routes stay public. Mutating routes require:

```text
Authorization: Bearer <token>
```

when the daemon is started with `--auth-token` or `MEMEX_AUTH_TOKEN`.

### `POST /upsert`

```bash
curl -X POST http://localhost:8997/upsert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MEMEX_AUTH_TOKEN" \
  -d '{
    "namespace": "memories",
    "id": "memory-001",
    "content": "Today I learned about Rust async...",
    "metadata": {"type": "learning", "date": "2026-01-13"}
  }'
```

### `POST /index`

```bash
curl -X POST http://localhost:8997/index \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MEMEX_AUTH_TOKEN" \
  -d '{
    "namespace": "docs",
    "content": "Long document text to be chunked...",
    "slice_mode": "onion",
    "metadata": {"source": "manual"}
  }'
```

## Dashboard Notes

The embedded dashboard now boots from `/api/discovery`, then drills into:

- `/api/browse` and `/api/browse/{ns}` for document lists
- `/search` for targeted queries

If the dashboard shows a loading state for a long time, inspect discovery directly:

```bash
curl http://localhost:8987/api/discovery
```

If discovery stays in `status = "loading"`, run:

```bash
rmcp-memex optimize
```

## Troubleshooting

### Dashboard Stuck In Loading

1. Check liveness: `curl http://localhost:8997/health`
2. Check discovery: `curl http://localhost:8987/api/discovery`
3. If `status` remains `"loading"`, run `rmcp-memex optimize`

### Port Conflict

Change the port in your config and restart:

```toml
# ~/.rmcp-servers/rmcp-memex/config.toml
http_port = 8997
```

### Network Exposure

If you bind the daemon beyond localhost, set `--auth-token` and explicit CORS origins.
