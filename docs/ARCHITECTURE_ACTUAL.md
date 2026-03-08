# Memex Architecture - ACTUAL (Stan Faktyczny)

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
        CODE["Code (.rs/.py/.js)<br/>❌ NO semantic chunking<br/>treated as plain text"]
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
        MLX_ACTUAL["MLX Embedder<br/>❌ Port: 12345 (not 8765)<br/>Model: Qwen3-Embedding-8B-4bit-DWQ<br/>Dims: 4096"]
        OLLAMA["Ollama (fallback)<br/>Port: 11434<br/>qwen3-embedding:8b"]
        BATCH["Smart Batching<br/>max 64 items<br/>max 128K chars"]
        NO_VALIDATE["❌ NO Dimension Validation<br/>Silent corruption possible"]
        RETRY["Retry with Backoff<br/>1s → 30s"]
    end

    subgraph STORAGE["LanceDB Storage"]
        RAMDISK["RAM Disk<br/>/Volumes/MemexRAM<br/>50GB HFS+"]
        LANCE["LanceDB<br/>~28GB vectors"]
        SCHEMA["Schema v3<br/>• id, namespace<br/>• vector[4096]<br/>• layer, parent_id<br/>• content_hash"]
        NO_ATOMIC["❌ NO Atomic Writes<br/>Partial failures = ghost docs"]
    end

    subgraph SEARCH["Search Pipeline"]
        ROUTER["Query Router<br/>auto-detect mode"]
        VECTOR["Vector Search<br/>ANN via IVF-HNSW"]
        BM25["BM25 Full-Text"]
        HYBRID["Hybrid Fusion<br/>RRF scoring"]
        NO_RERANK["❌ Reranker Optional<br/>Falls back to cosine"]
    end

    subgraph LAUNCHD["LaunchD Services"]
        LD_RAMDISK["ai.libraxis.memex-ramdisk<br/>Create 50GB RAM disk"]
        LD_MLX_ACTUAL["ai.libraxis.mlx-embedding<br/>❌ Port 12345 (not 8765)<br/>WorkDir: vista-brain/scripts"]
        LD_MEMEX["ai.libraxis.rmcp-memex<br/>Port 8987 server<br/>Uses RAM disk"]
        LD_SNAPSHOT["ai.libraxis.memex-snapshot<br/>Periodic sync"]
    end

    subgraph E2E_TESTS["E2E Tests"]
        NO_TEST_INDEX["❌ MISSING<br/>test_index_search"]
        NO_TEST_MCP["❌ MISSING<br/>test_mcp_tools"]
        NO_TEST_HTTP["❌ MISSING<br/>test_http_api"]
        NO_TEST_DEDUP["❌ MISSING<br/>test_deduplication"]
    end

    subgraph CONFIG["Configuration"]
        NO_CONFIG["❌ NO config.toml<br/>Uses env vars + hardcoded defaults"]
        NO_MLX_FOLDER["❌ ~/.ai-memories/mlx-embeddings/<br/>Contains unrelated project"]
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
    LD_MLX_ACTUAL -->|provides| MLX_ACTUAL
    LD_MEMEX -->|uses| RAMDISK
    LD_MEMEX -.->|should call| MLX_ACTUAL
    LD_SNAPSHOT -->|syncs| RAMDISK

    %% Styling - Green = works, Red = broken/missing, Yellow = partial
    classDef works fill:#2d5a2d,stroke:#4a4,color:#fff
    classDef broken fill:#5a2d2d,stroke:#a44,color:#fff
    classDef partial fill:#5a5a2d,stroke:#aa4,color:#fff
    classDef missing fill:#3d3d3d,stroke:#666,color:#888

    class PATH,TOKEN,TXT,PDF,JSON_SMART,MD,HASH,CHECK,SKIP,FLAT,ONION,FAST,BATCH,RETRY,RAMDISK,LANCE,SCHEMA,ROUTER,VECTOR,BM25,HYBRID,LD_RAMDISK,LD_MEMEX,LD_SNAPSHOT works
    class NO_VALIDATE,NO_ATOMIC,NO_RERANK,MLX_ACTUAL,LD_MLX_ACTUAL,NO_CONFIG,NO_MLX_FOLDER broken
    class CODE,OLLAMA partial
    class NO_TEST_INDEX,NO_TEST_MCP,NO_TEST_HTTP,NO_TEST_DEDUP missing
```

## Status: Co działa, co nie

### ✅ DZIAŁA (zielone)
| Komponent | Status | Uwagi |
|-----------|--------|-------|
| Path Validation | ✅ | Pełna walidacja traversal |
| Namespace Tokens | ✅ | create/revoke/verify |
| Plain Text extraction | ✅ | UTF-8 read |
| PDF extraction | ✅ | pdf_extract crate |
| Smart JSON detection | ✅ | Claude/ChatGPT/Session |
| Markdown extraction | ✅ | Section-aware |
| Deduplication | ✅ | SHA256 + storage check |
| Flat/Onion/Fast slicing | ✅ | Wszystkie 3 tryby |
| Smart Batching | ✅ | 64 items / 128K chars |
| Retry with Backoff | ✅ | 1s → 30s |
| RAM Disk | ✅ | 50GB /Volumes/MemexRAM |
| LanceDB | ✅ | 28GB, schema v3 |
| Query Router | ✅ | auto-detect |
| Vector/BM25/Hybrid search | ✅ | Wszystkie tryby |
| LaunchD services | ✅ | ramdisk, memex, snapshot |

### ❌ NIE DZIAŁA / NIE SPIĘTE (czerwone)
| Komponent | Problem | Impact |
|-----------|---------|--------|
| **MLX Port w kodzie** | Kod domyślnie 12345, serwer na **8765** | Config mismatch |
| **Atomic Writes** | BRAK - ghost docs przy crash | **HIGH** |
| **Reranker** | Optional, fallback to cosine | Słabsze wyniki |
| **config.toml** | BRAK - hardcoded defaults | Trudne zarządzanie |

### ✅ NAPRAWIONE (w tej sesji)
| Komponent | Status | Uwagi |
|-----------|--------|-------|
| **Dimension Validation** | ✅ DODANE | `test_dimension()` w `EmbeddingClient::new()` |
| **E2E Tests** | ✅ DODANE | `tests/e2e_pipeline.rs` - 5 testów |
| **TextIntegrityMetrics** | ✅ DODANE | >90% threshold, audit command |
| **DimensionAdapter** | ✅ DODANE | Cross-dim search 1024/2048/4096 |
| **Audit/Purge commands** | ✅ DODANE | `rust-memex audit`, `purge-quality` |

### ⚠️ CZĘŚCIOWE (żółte)
| Komponent | Status | Uwagi |
|-----------|--------|-------|
| Code extraction | ⚠️ | Traktowane jako plain text, brak AST |
| Ollama | ⚠️ | Działa jako fallback, ale to nie docelowy design |

### ❌ CAŁKOWICIE BRAKUJE (szare)
| Komponent | Status |
|-----------|--------|
| E2E test: index → search | ❌ BRAK |
| E2E test: MCP tools | ❌ BRAK |
| E2E test: HTTP API | ❌ BRAK |
| E2E test: deduplication | ❌ BRAK |

---

## Szczegóły LaunchD Services

### Aktualny stan usług:
```
PID     STATUS  SERVICE
10721   -15     ai.libraxis.rmcp-memex      ← działa, port 8987
46656   137     ai.libraxis.mlx-embedding   ← działa, port 12345 (!)
46670   137     ai.libraxis.mlx-reranker    ← działa
46743   1       ai.libraxis.mlx-batch-server
-       0       ai.libraxis.memex-snapshot
-       1       ai.libraxis.mlx-batch-runner
```

### ai.libraxis.mlx-embedding
```
Port:       12345 (❌ powinien być 8765)
WorkDir:    vista-brain/scripts/ (❌ powinien być ~/.ai-memories/mlx-embeddings/)
Script:     mlx_embedding_server.py
Model:      Qwen3-Embedding-8B-4bit-DWQ
Dims:       4096 ✅
```

### ai.libraxis.rmcp-memex
```
Port:       8987 ✅
DB Path:    /Volumes/MemexRAM/lancedb ✅
Mode:       --http-only ✅
PathState:  /Volumes/MemexRAM/lancedb (czeka na RAM disk) ✅
```

### ai.libraxis.memex-ramdisk
```
Size:       50GB (104857600 sectors)
Mount:      /Volumes/MemexRAM ✅
Source:     ~/.ai-memories/lancedb
Sync:       rsync -a ✅
```

---

## Krytyczne luki do naprawy

### 1. CRITICAL: Dimension Validation
```rust
// BRAK w embeddings/mod.rs
// Jeśli embedder zwróci 1024-dim:
// → Silent write to LanceDB
// → Cała baza corrupted
// → Brak recovery
```

### 2. HIGH: Atomic Batch Writes
```rust
// BRAK w storage/mod.rs
// Jeśli crash w połowie batch:
// → Ghost documents
// → Dedup nie złapie (inny hash?)
// → Brak rollback
```

### 3. MEDIUM: Port Mismatch
```
OBIECANE:   MLX embedder na 8765
FAKTYCZNE:  MLX embedder na 12345

rmcp-memex używa provider cascade:
1. Ollama localhost:11434
2. Fallback dragon:12345

Więc działa, ale przez Ollama, nie przez dedykowany MLX!
```

### 4. MEDIUM: Missing Config
```
OBIECANE:   ~/.ai-memories/config.toml
FAKTYCZNE:  Brak pliku

Wszystko przez env vars lub hardcoded defaults.
Trudne do zarządzania na wielu maszynach.
```

---

## Propozycja naprawy (kolejność priorytetów)

1. **[CRITICAL]** Dodać dimension validation w `EmbeddingClient::new()`
2. **[HIGH]** Implementować batch transaction z rollback
3. **[MEDIUM]** Zmienić port MLX na 8765 lub zaktualizować config memex
4. **[MEDIUM]** Stworzyć `~/.ai-memories/config.toml` z pełną konfiguracją
5. **[LOW]** Napisać testy E2E
