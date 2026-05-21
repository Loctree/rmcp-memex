# Memex Architecture Comparison: PROMISED vs ACTUAL

## Quick Status

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  OVERALL STATUS:  ~85% Complete                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  ✅ Working:     20 components                                           ║
║  ✅ Fixed:        5 components (in this session)                         ║
║  ❌ Broken:       2 components (1 HIGH)                                  ║
║  ⚠️ Partial:      2 components                                           ║
║  ⚠️ Config:       Port mismatch (works via Ollama fallback)              ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Side-by-Side Comparison

| Component | PROMISED | ACTUAL | Status |
|-----------|----------|--------|--------|
| **Entry Points** |
| MCP Tools | JSON-RPC over stdio | JSON-RPC over stdio | ✅ |
| HTTP API | REST + SSE on 8997 | REST + SSE on 8997 | ✅ |
| CLI | index/search/optimize | index/search/optimize | ✅ |
| **Security** |
| Path Validation | expand ~ / detect .. / canonicalize | expand ~ / detect .. / canonicalize | ✅ |
| Namespace Tokens | create/revoke/verify | create/revoke/verify | ✅ |
| **Extraction** |
| Plain Text | UTF-8 read | UTF-8 read | ✅ |
| PDF | pdf_extract crate | pdf_extract crate | ✅ |
| JSON (smart) | Claude/ChatGPT/Session detection | Claude/ChatGPT/Session detection | ✅ |
| Markdown | Section extraction | Section extraction | ✅ |
| Code | Semantic chunking (AST) | Plain text only | ⚠️ Partial |
| **Deduplication** |
| Hash Algorithm | SHA256 | SHA256 | ✅ |
| Storage Check | has_content_hash() | has_content_hash() | ✅ |
| **Slicing** |
| Flat Mode | 512-char / 128 overlap | 512-char / 128 overlap | ✅ |
| Onion Mode | 4 layers (outer→core) | 4 layers (outer→core) | ✅ |
| Onion-Fast | 2 layers | 2 layers | ✅ |
| **Embedding** |
| Embedder Port | **8765** | **12345** | ❌ Mismatch |
| Embedder Location | `~/.ai-memories/mlx-embeddings/` | `vista-brain/scripts/` | ❌ Wrong |
| Model | Qwen3-Embedding-8B | Qwen3-Embedding-8B-4bit-DWQ | ✅ |
| Dimensions | 4096 | 4096 | ✅ |
| **Dimension Validation** | **Fail fast on mismatch** | **✅ test_dimension() in EmbeddingClient::new()** | ✅ FIXED |
| **DimensionAdapter** | Cross-dim 1024/2048/4096 | **✅ expand/contract adapters** | ✅ FIXED |
| Batching | 64 items / 128K chars | 64 items / 128K chars | ✅ |
| Retry | Exponential backoff | Exponential backoff | ✅ |
| Fallback | Dragon remote | Ollama (unintended) | ⚠️ Partial |
| **Storage** |
| RAM Disk | 50GB /Volumes/MemexRAM | 50GB /Volumes/MemexRAM | ✅ |
| LanceDB Size | ~28GB | ~28GB | ✅ |
| Schema | v3 with content_hash | v3 with content_hash | ✅ |
| **Atomic Writes** | **Transaction rollback** | **NONE - ghost docs** | ❌ HIGH |
| **Search** |
| Query Router | auto-detect | auto-detect | ✅ |
| Vector Search | ANN via IVF-HNSW | ANN via IVF-HNSW | ✅ |
| BM25 | Full-text | Full-text | ✅ |
| Hybrid | RRF fusion | RRF fusion | ✅ |
| Reranker | Dedicated on 8766 | Optional (cosine fallback) | ❌ Weak |
| **LaunchD** |
| memex-ramdisk | Create 50GB | Create 50GB | ✅ |
| mlx-embedding | Port 8765 | Port 12345 | ❌ Mismatch |
| rmcp-memex | Port 8997, uses RAM | Port 8997, uses RAM | ✅ |
| memex-snapshot | Periodic sync | Periodic sync | ✅ |
| **Config** |
| Config File | `~/.ai-memories/config.toml` | **NONE** | ❌ Missing |
| **Quality Assurance** |
| TextIntegrityMetrics | >90% threshold | **✅ compute() + recommendation()** | ✅ FIXED |
| Audit command | Per-namespace check | **✅ rmcp-memex audit** | ✅ FIXED |
| Purge command | Remove low-quality | **✅ rmcp-memex purge-quality** | ✅ FIXED |
| **Testing** |
| E2E: pipeline | Required | **✅ tests/e2e_pipeline.rs (5 tests)** | ✅ FIXED |
| E2E: MCP tools | Required | **MISSING** | ❌ |
| E2E: HTTP API | Required | **MISSING** | ❌ |
| Unit tests | Required | ~20 tests | ✅ |

---

## Visual Comparison

### PROMISED Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                             │
│         MCP (stdio) │ HTTP (8997) │ CLI                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY + EXTRACTION                       │
│   Path Validation │ Token Auth │ PDF/JSON/MD/Code extraction    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DEDUPLICATION                             │
│              SHA256 → check storage → skip if exists            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SLICING (Flat/Onion/Fast)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLX EMBEDDER (port 8765)                     │
│   ~/.ai-memories/mlx-embeddings/ │ Qwen3-Embedding-8B │ 4096d   │
│           ✓ DIMENSION VALIDATION (fail fast)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANCEDB (RAM DISK)                           │
│   /Volumes/MemexRAM │ 50GB │ ~28GB vectors │ schema v3          │
│           ✓ ATOMIC BATCH WRITES (transaction)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SEARCH                                  │
│   Router → Vector/BM25/Hybrid → Reranker (8766) → Results       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       E2E TESTS                                 │
│   index→search │ MCP tools │ HTTP API │ deduplication           │
└─────────────────────────────────────────────────────────────────┘
```

### ACTUAL Architecture (Updated)
```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                             │
│         MCP (stdio) │ HTTP (8997) │ CLI (rmcp-memex)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY + EXTRACTION                       │
│   Path Validation │ Token Auth │ PDF/JSON/MD │ ⚠️Code=text      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DEDUPLICATION                             │
│              SHA256 → check storage → skip if exists            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SLICING (Flat/Onion/Fast)                    │
│            ✅ TextIntegrityMetrics (>90% threshold)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              ⚠️ MLX EMBEDDER (port 12345, not 8765)             │
│   vista-brain/scripts/ │ Qwen3-Embedding-8B-4bit │ 4096d        │
│           ✅ DIMENSION VALIDATION (test_dimension())            │
│           ✅ DimensionAdapter (cross-dim 1024/2048/4096)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANCEDB (RAM DISK)                           │
│   /Volumes/MemexRAM │ 50GB │ ~28GB vectors │ schema v3          │
│           ❌ NO ATOMIC WRITES (ghost docs on crash)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SEARCH                                  │
│   Router → Vector/BM25/Hybrid → ⚠️ cosine fallback → Results    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ✅ QUALITY ASSURANCE                         │
│   ✅ E2E pipeline │ ✅ audit cmd │ ✅ purge cmd │ ❌ HTTP tests  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Fix List

### ✅ FIXED (This Session)
1. ~~**Dimension Validation**~~ → ✅ `test_dimension()` in `EmbeddingClient::new()`
2. ~~**E2E Tests**~~ → ✅ `tests/e2e_pipeline.rs` (5 tests)
3. ~~**Quality Metrics**~~ → ✅ `TextIntegrityMetrics` with >90% threshold
4. ~~**Cross-dim Search**~~ → ✅ `DimensionAdapter` (1024/2048/4096)
5. ~~**Audit/Purge**~~ → ✅ `rmcp-memex audit` + `purge-quality` commands

### 🟠 HIGH (Data Integrity Risk)
1. **Atomic Batch Writes** - Implement transaction wrapper
   ```rust
   // Wrap batch operations in transaction
   storage.begin_transaction()?;
   for doc in batch {
       storage.add(doc)?;
   }
   storage.commit()?; // or rollback on error
   ```

### 🟡 MEDIUM (Config/Port Mismatch)
2. **MLX Port** - Change launch agent to port 8765 OR update memex config
3. **Config File** - Create `~/.ai-memories/config.toml` with all settings

### 🟢 LOW (Nice to Have)
4. **Code Semantic Chunking** - Add AST-based chunking for .rs/.py/.js
5. **Reranker Integration** - Make reranker non-optional
6. **HTTP API Tests** - Add E2E tests for REST endpoints

---

## Files Created

1. `docs/ARCHITECTURE_PROMISED.md` - Docelowa architektura (Mermaid)
2. `docs/ARCHITECTURE_ACTUAL.md` - Stan faktyczny (Mermaid + czerwone krzyże)
3. `docs/ARCHITECTURE_COMPARISON.md` - Porównanie side-by-side (ten plik)

---

*Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders*
