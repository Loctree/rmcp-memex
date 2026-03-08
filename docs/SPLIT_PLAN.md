# Plan: Rozdzielenie binariów CLI vs MCP Server

## Obecny stan
Jeden plik `src/bin/rmcp_memex.rs` (181KB, ~5300 LOC) zawiera wszystko.

## Docelowy podział

### 1. `rust-memex` (aliasy: `rmmx`, `rmemex`) - CLI
```
Komendy:
├── index      - Batch index documents
├── search     - Semantic search
├── dive       - Deep exploration (onion layers)
├── overview   - Stats and health
├── recall     - Memory recall with summary
├── timeline   - Show indexed content timeline
├── expand     - Drill down onion hierarchy
├── get        - Get chunk by ID
├── rag-search - RAG search across namespaces
├── namespaces - List namespaces
├── export     - Export to JSONL
├── upsert     - Upsert text chunk
├── optimize   - Compact + cleanup
├── health     - Health check
└── wizard     - Interactive config
```

### 2. `rmcp-memex` - MCP Server only
```
Tryby:
├── stdio      - JSON-RPC over stdio (default)
├── sse        - Server-Sent Events
└── http       - HTTP REST API

Flags:
├── --http-port PORT   - Enable HTTP/SSE
├── --http-only        - No stdio, only HTTP
└── --config PATH      - Config file
```

## Nowe pliki

```
src/bin/
├── rust_memex.rs      # CLI binary (renamed from rmcp_memex.rs, minus Serve)
└── rmcp_memex.rs      # MCP server only (Serve + HTTP handling)
```

## Cargo.toml changes

```toml
[[bin]]
name = "rust-memex"
path = "src/bin/rust_memex.rs"
required-features = ["cli"]

[[bin]]
name = "rmmx"           # Short alias
path = "src/bin/rust_memex.rs"
required-features = ["cli"]

[[bin]]
name = "rmcp-memex"
path = "src/bin/rmcp_memex.rs"
# No required-features - server can run without CLI deps
```

## Cross-dimension search (1024/2048/4096)

### Problem
Różne modele mają różne wymiary:
- nomic-embed-text: 768
- bge-large: 1024
- e5-large: 1024
- qwen3-embedding-8b: 4096

### Rozwiązanie: Dimension Adapter

```rust
pub struct DimensionAdapter {
    source_dim: usize,
    target_dim: usize,
}

impl DimensionAdapter {
    /// Pad smaller embeddings to target dimension
    pub fn expand(&self, embedding: Vec<f32>) -> Vec<f32> {
        if embedding.len() >= self.target_dim {
            return embedding[..self.target_dim].to_vec();
        }
        let mut padded = embedding;
        padded.resize(self.target_dim, 0.0);
        padded
    }

    /// Reduce larger embeddings to target dimension (PCA or truncate)
    pub fn contract(&self, embedding: Vec<f32>) -> Vec<f32> {
        // Simple truncation (or could use PCA projection)
        embedding[..self.target_dim].to_vec()
    }
}
```

### Cross-search implementation

```rust
pub async fn cross_dimension_search(
    query_embedding: Vec<f32>,
    target_dim: usize,
    storage: &StorageManager,
) -> Result<Vec<SearchResult>> {
    let adapter = DimensionAdapter::new(query_embedding.len(), target_dim);
    let adapted_query = adapter.adapt(query_embedding);

    // Search with adapted query
    storage.search_store(namespace, adapted_query, k).await
}
```

## Embedding Quality Metrics

### Text Integrity Score (>90% target)

```rust
pub struct TextIntegrityMetrics {
    /// Percentage of complete sentences preserved
    pub sentence_integrity: f32,

    /// Percentage of complete words (not truncated)
    pub word_integrity: f32,

    /// Average chunk length vs optimal
    pub chunk_quality: f32,

    /// Combined score
    pub overall: f32,
}

impl TextIntegrityMetrics {
    pub fn compute(original: &str, chunks: &[String]) -> Self {
        let original_sentences = count_sentences(original);
        let preserved_sentences = chunks.iter()
            .map(|c| count_complete_sentences(c))
            .sum::<usize>();

        let sentence_integrity = preserved_sentences as f32 / original_sentences as f32;

        // Check word boundaries
        let truncated_words = chunks.iter()
            .filter(|c| !ends_at_word_boundary(c))
            .count();
        let word_integrity = 1.0 - (truncated_words as f32 / chunks.len() as f32);

        let overall = (sentence_integrity + word_integrity) / 2.0;

        Self {
            sentence_integrity,
            word_integrity,
            chunk_quality: 1.0, // TODO
            overall,
        }
    }

    pub fn passes_threshold(&self) -> bool {
        self.overall >= 0.90
    }
}
```

### Quality check before indexing

```rust
pub async fn index_with_quality_check(
    content: &str,
    namespace: &str,
) -> Result<IndexResult> {
    let chunks = chunk_text(content);
    let metrics = TextIntegrityMetrics::compute(content, &chunks);

    if !metrics.passes_threshold() {
        tracing::warn!(
            "Text integrity below threshold: {:.1}% (min 90%)",
            metrics.overall * 100.0
        );
        // Could return error or just warn
    }

    // Proceed with indexing
    self.index_chunks(chunks, namespace).await
}
```

## Database Quality Audit

### Identify low-quality databases

```bash
# Command to audit all namespaces
rust-memex audit --all

# Output:
# Namespace: klaudiusz-sessions
#   Documents: 12,345
#   Avg chunk length: 127 chars  ❌ (min 200)
#   Sentence integrity: 45%      ❌ (min 90%)
#   Recommendation: PURGE
#
# Namespace: memories
#   Documents: 5,678
#   Avg chunk length: 512 chars  ✅
#   Sentence integrity: 94%      ✅
#   Recommendation: KEEP
```

### Purge command

```bash
# Dry run
rust-memex purge --below-quality 90 --dry-run

# Execute
rust-memex purge --below-quality 90 --confirm
```

## Migration Steps

1. [ ] Create `src/bin/rust_memex.rs` with CLI commands
2. [ ] Slim down `src/bin/rmcp_memex.rs` to server only
3. [ ] Update Cargo.toml with both binaries
4. [ ] Add DimensionAdapter to embeddings module
5. [ ] Add TextIntegrityMetrics to preprocessing
6. [ ] Add `audit` and `purge` commands
7. [ ] Test both binaries work independently
8. [ ] Update docs and help text

---
*Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders*
