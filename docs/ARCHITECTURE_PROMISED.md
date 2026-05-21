# Memex Architecture - PROMISED (Docelowa)

```mermaid
flowchart TB
    subgraph ENTRY["Entry Points"]
        MCP["MCP Tools<br/>(stdio JSON-RPC)"]
        HTTP["HTTP API<br/>(REST + SSE)"]
        CLI["CLI<br/>(commands)"]
    end

    subgraph VALIDATION["Security Layer"]
        PATH["Path Validation<br/>• expand ~<br/>• detect ..<br/>• canonicalize<br/>• whitelist check"]
        TOKEN["Namespace Tokens<br/>• create/revoke<br/>• per-namespace auth"]
    end

    subgraph EXTRACTION["Content Extraction"]
        TXT["Plain Text (.txt)"]
        PDF["PDF (.pdf)<br/>pdf_extract"]
        JSON_SMART["Smart JSON Detection<br/>• Claude.ai export<br/>• ChatGPT export<br/>• Session essence<br/>• Generic array"]
        MD["Markdown (.md)<br/>section extraction"]
        CODE["Code (.rs/.py/.js)<br/>semantic chunking"]
    end

    subgraph DEDUP["Deduplication"]
        HASH["SHA256 Hash<br/>content_hash"]
        CHECK["Storage Check<br/>has_content_hash()"]
        SKIP["Skip if EXISTS"]
    end

    subgraph SLICING["Slicing Strategy"]
        FLAT["FLAT MODE<br/>512-char chunks<br/>128-char overlap"]
        ONION["ONION MODE (default)<br/>4 hierarchical layers:<br/>OUTER → MIDDLE → INNER → CORE"]
        FAST["ONION-FAST<br/>2 layers only<br/>OUTER ↔ CORE"]
    end

    subgraph EMBEDDING["Embedding Generation"]
        MLX["MLX Native Embedder<br/>Port: 8765<br/>Model: Qwen3-Embedding-8B<br/>Dims: 4096"]
        BATCH["Smart Batching<br/>max 64 items<br/>max 128K chars"]
        VALIDATE["Dimension Validation<br/>FAIL FAST if mismatch"]
        RETRY["Retry with Backoff<br/>1s → 30s"]
    end

    subgraph STORAGE["LanceDB Storage"]
        RAMDISK["RAM Disk<br/>/Volumes/MemexRAM<br/>50GB HFS+"]
        LANCE["LanceDB<br/>~28GB vectors"]
        SCHEMA["Schema v3<br/>• id, namespace<br/>• vector[4096]<br/>• layer, parent_id<br/>• content_hash"]
        ATOMIC["Atomic Batch Writes<br/>Transaction rollback"]
    end

    subgraph SEARCH["Search Pipeline"]
        ROUTER["Query Router<br/>auto-detect mode"]
        VECTOR["Vector Search<br/>ANN via IVF-HNSW"]
        BM25["BM25 Full-Text"]
        HYBRID["Hybrid Fusion<br/>RRF scoring"]
        RERANK["Reranker<br/>Port: 8766<br/>cross-encoder"]
    end

    subgraph LAUNCHD["LaunchD Services"]
        LD_RAMDISK["ai.libraxis.memex-ramdisk<br/>Create 50GB RAM disk"]
        LD_MLX["ai.libraxis.mlx-embedding<br/>Port 8765 embedder"]
        LD_MEMEX["ai.libraxis.rmcp-memex<br/>Port 8997 server"]
        LD_SNAPSHOT["ai.libraxis.memex-snapshot<br/>Periodic sync to disk"]
    end

    subgraph E2E_TESTS["E2E Tests"]
        TEST_INDEX["test_index_search<br/>File → Vector → Search"]
        TEST_MCP["test_mcp_tools<br/>JSON-RPC handlers"]
        TEST_HTTP["test_http_api<br/>REST endpoints"]
        TEST_DEDUP["test_deduplication<br/>Hash collision handling"]
    end

    %% Flow connections
    ENTRY --> VALIDATION
    VALIDATION --> EXTRACTION
    EXTRACTION --> DEDUP
    DEDUP -->|NEW| SLICING
    DEDUP -->|EXISTS| SKIP
    SLICING --> EMBEDDING
    EMBEDDING --> STORAGE

    %% Search flow
    ENTRY --> SEARCH
    SEARCH --> STORAGE

    %% LaunchD flow
    LD_RAMDISK -->|creates| RAMDISK
    LD_MLX -->|provides| MLX
    LD_MEMEX -->|uses| RAMDISK
    LD_MEMEX -->|calls| MLX
    LD_SNAPSHOT -->|syncs| RAMDISK

    %% Styling
    classDef promised fill:#2d5a2d,stroke:#4a4,color:#fff
    classDef critical fill:#5a2d2d,stroke:#a44,color:#fff

    class MLX,VALIDATE,ATOMIC,TEST_INDEX,TEST_MCP,TEST_HTTP,TEST_DEDUP,LD_MLX promised
    class RAMDISK,LANCE,SCHEMA critical
```

## Kluczowe założenia docelowej architektury

### 1. Embedding Layer
- **Dedykowany MLX embedder** na porcie **8765**
- Folder implementacji: `~/.ai-memories/mlx-embeddings/`
- Model: `Qwen3-Embedding-8B` (4096 dims)
- **NIE Ollama** - natywny MLX dla Dragona

### 2. RAM Disk Architecture
- 50GB RAM disk `/Volumes/MemexRAM`
- LanceDB (~28GB) całkowicie w RAM
- Snapshot daemon - periodic sync to `~/.ai-memories/lancedb`
- PathState dependency - memex startuje dopiero gdy RAM disk gotowy

### 3. Dimension Safety
- **Validation at startup** - sprawdzenie czy embedder zwraca 4096 dims
- **Fail fast** - crash jeśli mismatch, nie silent corruption

### 4. Atomic Writes
- Transaction boundaries na batch writes
- Rollback przy partial failures
- No ghost documents

### 5. E2E Test Coverage
- Full pipeline tests (index → embed → store → search → verify)
- MCP tool integration tests
- HTTP API endpoint tests
- Deduplication regression tests
