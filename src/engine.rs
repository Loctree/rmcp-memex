//! High-level MemexEngine API for library consumers.
//!
//! The `MemexEngine` provides a simple, ergonomic interface for storing and
//! searching vector embeddings. It wraps the lower-level `StorageManager` and
//! `EmbeddingClient` to provide a unified API.
//!
//! # Example
//!
//! ```rust,ignore
//! use rmcp_memex::{MemexEngine, MemexConfig};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Quick setup for an app
//!     let engine = MemexEngine::for_app("my-app", "documents").await?;
//!
//!     // Store a document
//!     engine.store("doc-1", "Hello world!", json!({"source": "test"})).await?;
//!
//!     // Search for similar documents
//!     let results = engine.search("greeting", 5).await?;
//!
//!     // Get by ID
//!     if let Some(doc) = engine.get("doc-1").await? {
//!         println!("Found: {}", doc.text);
//!     }
//!
//!     // Delete
//!     engine.delete("doc-1").await?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::embeddings::{DEFAULT_REQUIRED_DIMENSION, EmbeddingClient, EmbeddingConfig};
use crate::rag::{SearchOptions, SearchResult, SliceLayer};
use crate::search::{
    BM25Config, BM25Index, HybridConfig, HybridSearchResult, HybridSearcher, SearchMode,
};
use crate::storage::{ChromaDocument, StorageManager};

// Re-export SearchResult for convenience
pub use crate::rag::SearchResult as Document;

/// Configuration for MemexEngine.
///
/// Provides sensible defaults while allowing customization of all components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemexConfig {
    /// Application name (used for default db_path)
    pub app_name: String,
    /// Namespace for document isolation
    pub namespace: String,
    /// Path to LanceDB storage (defaults to ~/.rmcp-servers/{app_name}/lancedb)
    #[serde(default)]
    pub db_path: Option<String>,
    /// Embedding vector dimension (must match your embedding model)
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Embedding provider configuration
    #[serde(default)]
    pub embedding_config: EmbeddingConfig,
    /// Enable BM25 keyword search
    #[serde(default)]
    pub enable_bm25: bool,
    /// BM25 configuration (if enabled)
    #[serde(default)]
    pub bm25_config: Option<BM25Config>,
    /// Enable hybrid search (vector + BM25 fusion)
    #[serde(default = "default_enable_hybrid")]
    pub enable_hybrid: bool,
    /// Hybrid search configuration
    #[serde(default)]
    pub hybrid_config: Option<HybridConfig>,
}

fn default_enable_hybrid() -> bool {
    true // Hybrid enabled by default
}

fn default_dimension() -> usize {
    DEFAULT_REQUIRED_DIMENSION
}

impl Default for MemexConfig {
    fn default() -> Self {
        Self {
            app_name: "memex".to_string(),
            namespace: "default".to_string(),
            db_path: None,
            dimension: default_dimension(),
            embedding_config: EmbeddingConfig::default(),
            enable_bm25: false,
            bm25_config: None,
            enable_hybrid: default_enable_hybrid(),
            hybrid_config: None,
        }
    }
}

impl MemexConfig {
    /// Create a new config for an app with a namespace
    pub fn new(app_name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            namespace: namespace.into(),
            ..Default::default()
        }
    }

    /// Set custom database path
    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    /// Set embedding dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self.embedding_config.required_dimension = dimension;
        self
    }

    /// Set embedding configuration
    pub fn with_embedding_config(mut self, config: EmbeddingConfig) -> Self {
        self.dimension = config.required_dimension;
        self.embedding_config = config;
        self
    }

    fn sync_dimension_fields(&mut self) -> Result<()> {
        if self.dimension == self.embedding_config.required_dimension {
            return Ok(());
        }

        let default_dim = default_dimension();
        if self.dimension == default_dim {
            self.dimension = self.embedding_config.required_dimension;
            return Ok(());
        }

        if self.embedding_config.required_dimension == default_dim {
            self.embedding_config.required_dimension = self.dimension;
            return Ok(());
        }

        Err(anyhow!(
            "MemexConfig.dimension={} conflicts with embedding_config.required_dimension={}. \
             Set them to the same value or use with_dimension()/with_embedding_config() so one source of truth updates both.",
            self.dimension,
            self.embedding_config.required_dimension
        ))
    }

    /// Enable BM25 hybrid search
    pub fn with_bm25(mut self, config: BM25Config) -> Self {
        self.enable_bm25 = true;
        self.bm25_config = Some(config);
        self
    }

    /// Get the effective database path
    pub fn effective_db_path(&self) -> String {
        self.db_path
            .clone()
            .unwrap_or_else(|| format!("~/.rmcp-servers/{}/lancedb", self.app_name))
    }

    /// Get the effective BM25 path
    pub fn effective_bm25_path(&self) -> String {
        self.bm25_config
            .as_ref()
            .map(|c| c.index_path.clone())
            .unwrap_or_else(|| format!("~/.rmcp-servers/{}/bm25", self.app_name))
    }

    fn hybrid_uses_bm25(&self) -> bool {
        self.enable_hybrid
            && self.hybrid_config.clone().unwrap_or_default().mode != SearchMode::Vector
    }

    fn normalize_bm25_config(&self, mut config: BM25Config) -> BM25Config {
        if config.index_path == BM25Config::default().index_path {
            config.index_path = self.effective_bm25_path();
        }
        config
    }

    fn resolved_bm25_config(&self) -> Option<BM25Config> {
        if !self.enable_bm25 && !self.hybrid_uses_bm25() {
            return None;
        }

        let config = self
            .bm25_config
            .clone()
            .or_else(|| {
                self.hybrid_config
                    .as_ref()
                    .filter(|cfg| cfg.mode != SearchMode::Vector)
                    .map(|cfg| cfg.bm25.clone())
            })
            .unwrap_or_default();

        Some(self.normalize_bm25_config(config))
    }

    fn resolved_hybrid_config(&self) -> HybridConfig {
        let mut config = self.hybrid_config.clone().unwrap_or_default();
        if let Some(bm25) = self.resolved_bm25_config() {
            config.bm25 = bm25;
        }
        config
    }
}

/// Metadata filter for search and deletion operations.
///
/// Used for filtering documents by metadata fields (e.g., patient_id, visit_id).
/// Supports GDPR-compliant data deletion by patient.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaFilter {
    /// Filter by patient ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patient_id: Option<String>,
    /// Filter by visit ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visit_id: Option<String>,
    /// Filter by document type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_type: Option<String>,
    /// Filter by date range (start)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_from: Option<String>,
    /// Filter by date range (end)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_to: Option<String>,
    /// Custom metadata key-value filters
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub custom: Vec<(String, String)>,
}

impl MetaFilter {
    /// Create a filter for a specific patient (GDPR deletion use case)
    pub fn for_patient(patient_id: impl Into<String>) -> Self {
        Self {
            patient_id: Some(patient_id.into()),
            ..Default::default()
        }
    }

    /// Create a filter for a specific visit
    pub fn for_visit(visit_id: impl Into<String>) -> Self {
        Self {
            visit_id: Some(visit_id.into()),
            ..Default::default()
        }
    }

    /// Add a custom metadata filter
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.push((key.into(), value.into()));
        self
    }

    /// Check if this filter matches a document's metadata
    pub fn matches(&self, metadata: &Value) -> bool {
        if let Some(ref patient_id) = self.patient_id
            && metadata.get("patient_id").and_then(|v| v.as_str()) != Some(patient_id)
        {
            return false;
        }

        if let Some(ref visit_id) = self.visit_id
            && metadata.get("visit_id").and_then(|v| v.as_str()) != Some(visit_id)
        {
            return false;
        }

        if let Some(ref doc_type) = self.doc_type
            && metadata.get("doc_type").and_then(|v| v.as_str()) != Some(doc_type)
        {
            return false;
        }

        // Date range filtering
        if let Some(ref date_from) = self.date_from
            && let Some(doc_date) = metadata.get("date").and_then(|v| v.as_str())
            && doc_date < date_from.as_str()
        {
            return false;
        }

        if let Some(ref date_to) = self.date_to
            && let Some(doc_date) = metadata.get("date").and_then(|v| v.as_str())
            && doc_date > date_to.as_str()
        {
            return false;
        }

        // Custom filters
        for (key, value) in &self.custom {
            if metadata.get(key).and_then(|v| v.as_str()) != Some(value) {
                return false;
            }
        }

        true
    }
}

/// Item for batch storage operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreItem {
    /// Unique document ID
    pub id: String,
    /// Text content to embed and store
    pub text: String,
    /// Optional metadata
    #[serde(default)]
    pub metadata: Value,
}

impl StoreItem {
    /// Create a new store item
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Value::Object(serde_json::Map::new()),
        }
    }

    /// Add metadata to this item
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Result of a batch operation
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of items successfully processed
    pub success_count: usize,
    /// Number of items that failed
    pub failure_count: usize,
    /// IDs of failed items (if any)
    pub failed_ids: Vec<String>,
}

/// Statistics for a single layer in dive results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    /// Total number of chunks found in this layer
    pub total_chunks: usize,
    /// Average score of results in this layer
    pub avg_score: f32,
    /// Top keywords across results in this layer
    pub top_keywords: Vec<String>,
}

impl LayerStats {
    /// Create empty layer stats
    pub fn empty() -> Self {
        Self {
            total_chunks: 0,
            avg_score: 0.0,
            top_keywords: vec![],
        }
    }

    /// Create layer stats from search results
    pub fn from_results(results: &[SearchResult]) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let total_chunks = results.len();
        let avg_score = results.iter().map(|r| r.score).sum::<f32>() / total_chunks as f32;

        // Aggregate keywords across results
        let mut keyword_counts: HashMap<String, usize> = HashMap::new();
        for result in results {
            for keyword in &result.keywords {
                *keyword_counts.entry(keyword.clone()).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take top 10
        let mut keywords: Vec<_> = keyword_counts.into_iter().collect();
        keywords.sort_by_key(|b| std::cmp::Reverse(b.1));
        let top_keywords = keywords.into_iter().take(10).map(|(k, _)| k).collect();

        Self {
            total_chunks,
            avg_score,
            top_keywords,
        }
    }
}

/// Result of a dive operation for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiveResult {
    /// The layer this result is for
    pub layer: SliceLayer,
    /// Search results for this layer
    pub results: Vec<SearchResult>,
    /// Statistics for this layer
    pub layer_stats: LayerStats,
}

/// High-level API for vector memory operations.
///
/// MemexEngine provides a simple interface for storing, searching, and managing
/// vector embeddings. It orchestrates the embedding client and storage manager.
pub struct MemexEngine {
    storage: Arc<StorageManager>,
    embeddings: Arc<Mutex<EmbeddingClient>>,
    bm25: Option<Arc<BM25Index>>,
    hybrid_searcher: Option<HybridSearcher>,
    namespace: String,
    config: MemexConfig,
}

impl MemexEngine {
    /// Create a new MemexEngine with the given configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = MemexConfig::new("my-app", "documents")
    ///     .with_dimension(1024);
    /// let engine = MemexEngine::new(config).await?;
    /// ```
    pub async fn new(mut config: MemexConfig) -> Result<Self> {
        config.sync_dimension_fields()?;
        let db_path = config.effective_db_path();

        info!(
            "Initializing MemexEngine: app={}, namespace={}, db={}",
            config.app_name, config.namespace, db_path
        );

        // Initialize storage
        let storage = StorageManager::new_lance_only(&db_path).await?;
        storage.ensure_collection().await?;

        // Initialize embedding client
        let embeddings = EmbeddingClient::new(&config.embedding_config).await?;

        info!(
            "Connected to embedding provider: {} (dim={})",
            embeddings.connected_to(),
            embeddings.required_dimension()
        );

        // Initialize BM25 if enabled
        let bm25 = config
            .resolved_bm25_config()
            .map(|bm25_config| BM25Index::new(&bm25_config).map(Arc::new))
            .transpose()?;

        let storage_arc = Arc::new(storage);

        // Initialize HybridSearcher if hybrid mode is enabled
        let hybrid_searcher = if config.enable_hybrid {
            let hybrid_config = config.resolved_hybrid_config();
            Some(if let Some(ref bm25_index) = bm25 {
                HybridSearcher::with_bm25_index(
                    storage_arc.clone(),
                    bm25_index.clone(),
                    hybrid_config,
                )
            } else {
                HybridSearcher::new(storage_arc.clone(), hybrid_config).await?
            })
        } else {
            None
        };

        Ok(Self {
            storage: storage_arc,
            embeddings: Arc::new(Mutex::new(embeddings)),
            bm25,
            hybrid_searcher,
            namespace: config.namespace.clone(),
            config,
        })
    }

    /// Quick setup for an application.
    ///
    /// Uses default embedding configuration and auto-detects providers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let engine = MemexEngine::for_app("vista", "patient-notes").await?;
    /// ```
    pub async fn for_app(app_name: &str, namespace: &str) -> Result<Self> {
        let config = MemexConfig::new(app_name, namespace);
        Self::new(config).await
    }

    /// Vista-optimized setup with 1024-dimension embeddings.
    ///
    /// Uses smaller embedding model (qwen3-embedding:0.6b) for faster inference.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let engine = MemexEngine::for_vista().await?;
    /// ```
    pub async fn for_vista() -> Result<Self> {
        use crate::embeddings::ProviderConfig;

        let config = MemexConfig {
            app_name: "vista".to_string(),
            namespace: "default".to_string(),
            db_path: Some("~/.rmcp-servers/vista/lancedb".to_string()),
            dimension: 1024,
            embedding_config: EmbeddingConfig {
                required_dimension: 1024,
                providers: vec![ProviderConfig {
                    name: "ollama-vista".to_string(),
                    base_url: "http://localhost:11434".to_string(),
                    model: "qwen3-embedding:0.6b".to_string(),
                    priority: 1,
                    endpoint: "/v1/embeddings".to_string(),
                }],
                ..EmbeddingConfig::default()
            },
            enable_bm25: false,
            bm25_config: None,
            enable_hybrid: true, // Hybrid enabled for Vista
            hybrid_config: None,
        };
        Self::new(config).await
    }

    /// Get the namespace this engine operates on
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the configuration
    pub fn config(&self) -> &MemexConfig {
        &self.config
    }

    /// Get the underlying storage manager (for advanced operations)
    pub fn storage(&self) -> Arc<StorageManager> {
        self.storage.clone()
    }

    // =========================================================================
    // CORE CRUD OPERATIONS
    // =========================================================================

    /// Store a document with embedding.
    ///
    /// The text is automatically embedded using the configured embedding provider.
    ///
    /// # Arguments
    /// * `id` - Unique document identifier
    /// * `text` - Text content to embed and store
    /// * `metadata` - Additional metadata (JSON object)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// engine.store(
    ///     "visit-123",
    ///     "Patient presented with lethargy and decreased appetite...",
    ///     json!({"patient_id": "P-456", "visit_type": "checkup"})
    /// ).await?;
    /// ```
    pub async fn store(&self, id: &str, text: &str, metadata: Value) -> Result<()> {
        debug!("Storing document: id={}, text_len={}", id, text.len());

        // Generate embedding
        let embedding = self.embeddings.lock().await.embed(text).await?;

        // Create document
        let doc = ChromaDocument::new_flat(
            id.to_string(),
            self.namespace.clone(),
            embedding,
            metadata.clone(),
            text.to_string(),
        );

        // Store in vector DB
        self.storage.add_to_store(vec![doc]).await?;

        // Also index in BM25 if enabled
        if let Some(ref bm25) = self.bm25 {
            bm25.add_documents(&[(id.to_string(), self.namespace.clone(), text.to_string())])
                .await?;
        }

        debug!("Stored document: id={}", id);
        Ok(())
    }

    /// Search for similar documents.
    ///
    /// Returns documents ordered by similarity score (highest first).
    ///
    /// # Arguments
    /// * `query` - Search query text
    /// * `limit` - Maximum number of results
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = engine.search("lethargy symptoms", 10).await?;
    /// for result in results {
    ///     println!("{}: {} (score: {})", result.id, result.text, result.score);
    /// }
    /// ```
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Searching: query='{}', limit={}", query, limit);

        // Generate query embedding
        let query_embedding = self.embeddings.lock().await.embed(query).await?;

        // Search vector store
        let candidates = self
            .storage
            .search_store(Some(&self.namespace), query_embedding, limit)
            .await?;

        // Convert to SearchResult
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, doc)| {
                // Simple inverse-index scoring (better results have lower index)
                let score = 1.0 - (idx as f32 / (limit as f32 + 1.0));
                let layer = doc.slice_layer();
                SearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    text: doc.document,
                    score,
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                }
            })
            .collect();

        debug!("Search returned {} results", results.len());
        Ok(results)
    }

    /// Hybrid search combining vector similarity and BM25 keyword matching.
    ///
    /// Returns results with combined scores from both methods.
    /// Requires `enable_hybrid: true` in config.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = engine.search_hybrid("when did we buy dragon", 10).await?;
    /// for r in results {
    ///     println!("{}: combined={:.3}, vector={:?}, bm25={:?}",
    ///         r.id, r.combined_score, r.vector_score, r.bm25_score);
    /// }
    /// ```
    pub async fn search_hybrid(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        debug!("Hybrid search: query='{}', limit={}", query, limit);

        let hybrid = self.hybrid_searcher.as_ref().ok_or_else(|| {
            anyhow!("Hybrid search not enabled. Set enable_hybrid: true in MemexConfig.")
        })?;

        // Generate query embedding
        let query_embedding = self.embeddings.lock().await.embed(query).await?;

        // Perform hybrid search
        let results = hybrid
            .search(
                query,
                query_embedding,
                Some(&self.namespace),
                limit,
                SearchOptions::default(),
            )
            .await?;

        debug!("Hybrid search returned {} results", results.len());
        Ok(results)
    }

    /// Search with explicit mode selection.
    ///
    /// Allows choosing between vector-only, keyword-only, or hybrid search.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rmcp_memex::SearchMode;
    ///
    /// // Keyword-only for exact matches
    /// let results = engine.search_with_mode("dragon", 10, SearchMode::Keyword).await?;
    /// ```
    pub async fn search_with_mode(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<HybridSearchResult>> {
        debug!("Search with mode: query='{}', mode={:?}", query, mode);

        match mode {
            SearchMode::Vector => {
                // Use regular vector search and convert to HybridSearchResult
                let results = self.search(query, limit).await?;
                Ok(results
                    .into_iter()
                    .map(|r| HybridSearchResult {
                        id: r.id,
                        namespace: r.namespace,
                        document: r.text,
                        combined_score: r.score,
                        vector_score: Some(r.score),
                        bm25_score: None,
                        metadata: r.metadata,
                        layer: r.layer,
                        parent_id: r.parent_id,
                        children_ids: r.children_ids,
                        keywords: r.keywords,
                    })
                    .collect())
            }
            SearchMode::Keyword | SearchMode::Hybrid => {
                // Use hybrid searcher
                self.search_hybrid(query, limit).await
            }
        }
    }

    /// Get a document by ID.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(doc) = engine.get("visit-123").await? {
    ///     println!("Found: {}", doc.text);
    /// }
    /// ```
    pub async fn get(&self, id: &str) -> Result<Option<SearchResult>> {
        debug!("Getting document: id={}", id);

        if let Some(doc) = self.storage.get_document(&self.namespace, id).await? {
            let layer = doc.slice_layer();
            return Ok(Some(SearchResult {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document,
                score: 1.0,
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            }));
        }

        Ok(None)
    }

    /// Delete a document by ID.
    ///
    /// Returns true if a document was deleted, false if not found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if engine.delete("visit-123").await? {
    ///     println!("Document deleted");
    /// }
    /// ```
    pub async fn delete(&self, id: &str) -> Result<bool> {
        debug!("Deleting document: id={}", id);

        let deleted = self.storage.delete_document(&self.namespace, id).await?;

        // Also delete from BM25 if enabled
        if let Some(ref bm25) = self.bm25 {
            bm25.delete_documents(&[id.to_string()]).await?;
        }

        Ok(deleted > 0)
    }

    // =========================================================================
    // BATCH OPERATIONS
    // =========================================================================

    /// Store multiple documents in a batch.
    ///
    /// More efficient than calling `store()` multiple times as embeddings
    /// are generated in batches.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let items = vec![
    ///     StoreItem::new("doc-1", "First document").with_metadata(json!({"type": "note"})),
    ///     StoreItem::new("doc-2", "Second document").with_metadata(json!({"type": "note"})),
    /// ];
    /// let result = engine.store_batch(items).await?;
    /// println!("Stored {} documents", result.success_count);
    /// ```
    pub async fn store_batch(&self, items: Vec<StoreItem>) -> Result<BatchResult> {
        if items.is_empty() {
            return Ok(BatchResult {
                success_count: 0,
                failure_count: 0,
                failed_ids: vec![],
            });
        }

        info!("Batch storing {} documents", items.len());

        // Extract texts for batch embedding
        let texts: Vec<String> = items.iter().map(|i| i.text.clone()).collect();

        // Generate embeddings in batch
        let embeddings = self.embeddings.lock().await.embed_batch(&texts).await?;

        // Create documents
        let mut docs = Vec::with_capacity(items.len());
        let mut bm25_docs = Vec::new();

        for (item, embedding) in items.iter().zip(embeddings) {
            let doc = ChromaDocument::new_flat(
                item.id.clone(),
                self.namespace.clone(),
                embedding,
                item.metadata.clone(),
                item.text.clone(),
            );
            docs.push(doc);

            if self.bm25.is_some() {
                bm25_docs.push((item.id.clone(), self.namespace.clone(), item.text.clone()));
            }
        }

        // Store in vector DB
        self.storage.add_to_store(docs).await?;

        // Also index in BM25 if enabled
        if let Some(ref bm25) = self.bm25 {
            bm25.add_documents(&bm25_docs).await?;
        }

        Ok(BatchResult {
            success_count: items.len(),
            failure_count: 0,
            failed_ids: vec![],
        })
    }

    // =========================================================================
    // FILTERED OPERATIONS
    // =========================================================================

    /// Search with metadata filter.
    ///
    /// Performs vector search and then filters results by metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let filter = MetaFilter::for_patient("P-456");
    /// let results = engine.search_filtered("symptoms", filter, 10).await?;
    /// ```
    pub async fn search_filtered(
        &self,
        query: &str,
        filter: MetaFilter,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // Fetch more candidates than needed, then filter
        let candidates = self.search(query, limit * 3).await?;

        // Apply metadata filter
        let filtered: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(limit)
            .collect();

        debug!(
            "Filtered search: query='{}', filter={:?}, results={}",
            query,
            filter,
            filtered.len()
        );

        Ok(filtered)
    }

    /// Delete all documents matching a filter.
    ///
    /// This is the primary method for GDPR-compliant data deletion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Delete all documents for a patient (GDPR request)
    /// let filter = MetaFilter::for_patient("P-456");
    /// let deleted = engine.delete_by_filter(filter).await?;
    /// println!("Deleted {} documents", deleted);
    /// ```
    pub async fn delete_by_filter(&self, filter: MetaFilter) -> Result<usize> {
        info!("Deleting documents by filter: {:?}", filter);

        // We need to search for all matching documents first
        // This is expensive but necessary for metadata-based filtering
        // Note: A more efficient implementation would add filter support to StorageManager

        // For now, we'll scan namespace documents and filter in memory.
        // TODO: Add native metadata filtering to LanceDB queries.

        let mut deleted_ids = Vec::new();

        // Search namespace documents page by page before mutating the table.
        // Deleting while paginating would shift row offsets and risk skipping
        // matches in later pages.
        const BATCH_SIZE: usize = 1000;
        let mut offset = 0;

        loop {
            let candidates = self
                .storage
                .all_documents_page(Some(&self.namespace), offset, BATCH_SIZE)
                .await?;

            if candidates.is_empty() {
                break;
            }

            let page_len = candidates.len();
            for doc in candidates {
                if filter.matches(&doc.metadata) {
                    deleted_ids.push(doc.id);
                }
            }

            if page_len < BATCH_SIZE {
                break;
            }

            offset += page_len;
        }

        for id in &deleted_ids {
            self.storage.delete_document(&self.namespace, id).await?;
        }

        // Delete from BM25 if enabled
        if let Some(ref bm25) = self.bm25
            && !deleted_ids.is_empty()
        {
            bm25.delete_documents(&deleted_ids).await?;
        }

        let deleted_count = deleted_ids.len();
        info!("Deleted {} documents by filter", deleted_count);
        Ok(deleted_count)
    }

    /// Delete all documents in the namespace.
    ///
    /// Use with caution - this removes all data!
    pub async fn purge_namespace(&self) -> Result<usize> {
        info!("Purging namespace: {}", self.namespace);

        let deleted = self
            .storage
            .delete_namespace_documents(&self.namespace)
            .await?;

        if let Some(ref bm25) = self.bm25 {
            bm25.delete_namespace_term(&self.namespace).await?;
        }

        Ok(deleted)
    }

    // =========================================================================
    // HYBRID SEARCH (BM25 + Vector)
    // =========================================================================

    /// Hybrid search combining BM25 keyword matching with vector similarity.
    ///
    /// Requires `enable_bm25: true` in config.
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `limit` - Maximum results
    /// * `bm25_weight` - Weight for BM25 scores (0.0-1.0, default 0.3)
    #[deprecated(
        since = "0.3.1",
        note = "Use search_hybrid() with HybridSearcher instead"
    )]
    pub async fn search_bm25_fusion(
        &self,
        query: &str,
        limit: usize,
        bm25_weight: f32,
    ) -> Result<Vec<SearchResult>> {
        let bm25 = self
            .bm25
            .as_ref()
            .ok_or_else(|| anyhow!("BM25 not enabled. Set enable_bm25: true in MemexConfig."))?;

        // Get BM25 results
        let bm25_results = bm25.search(query, Some(&self.namespace), limit * 2)?;
        let bm25_max_score = bm25_results.first().map(|(_, _, s)| *s).unwrap_or(1.0);

        // Get vector results
        let vector_results = self.search(query, limit * 2).await?;

        // Merge and re-score
        use std::collections::HashMap;
        let mut scores: HashMap<String, (f32, Option<SearchResult>)> = HashMap::new();

        // Add BM25 scores (normalized)
        for (id, _namespace, score) in bm25_results {
            let normalized = score / bm25_max_score.max(0.001);
            scores.insert(id, (normalized * bm25_weight, None));
        }

        // Add vector scores
        let vector_weight = 1.0 - bm25_weight;
        for result in vector_results {
            let entry = scores.entry(result.id.clone()).or_insert((0.0, None));
            entry.0 += result.score * vector_weight;
            entry.1 = Some(result);
        }

        // Collect and sort by combined score
        let mut combined: Vec<_> = scores
            .into_iter()
            .filter_map(|(_id, (score, result))| {
                // If we have the full result, use it; otherwise fetch from storage
                result.map(|mut r| {
                    r.score = score;
                    r
                })
            })
            .collect();

        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        combined.truncate(limit);

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_filter_matches() {
        let filter = MetaFilter::for_patient("P-123");

        let matching = serde_json::json!({
            "patient_id": "P-123",
            "visit_id": "V-456"
        });
        assert!(filter.matches(&matching));

        let not_matching = serde_json::json!({
            "patient_id": "P-999",
            "visit_id": "V-456"
        });
        assert!(!filter.matches(&not_matching));
    }

    #[test]
    fn test_meta_filter_custom() {
        let filter = MetaFilter::default()
            .with_custom("doc_type", "soap_note")
            .with_custom("status", "active");

        let matching = serde_json::json!({
            "doc_type": "soap_note",
            "status": "active"
        });
        assert!(filter.matches(&matching));

        let missing_field = serde_json::json!({
            "doc_type": "soap_note"
        });
        assert!(!filter.matches(&missing_field));
    }

    #[test]
    fn test_memex_config_defaults() {
        let config = MemexConfig::default();
        assert_eq!(config.dimension, DEFAULT_REQUIRED_DIMENSION);
        assert_eq!(
            config.embedding_config.required_dimension,
            DEFAULT_REQUIRED_DIMENSION
        );
        assert_eq!(config.namespace, "default");
        assert_eq!(config.effective_db_path(), "~/.rmcp-servers/memex/lancedb");
    }

    #[test]
    fn test_memex_config_builder() {
        let config = MemexConfig::new("vista", "patients")
            .with_dimension(1024)
            .with_db_path("/custom/path/db");

        assert_eq!(config.app_name, "vista");
        assert_eq!(config.namespace, "patients");
        assert_eq!(config.dimension, 1024);
        assert_eq!(config.embedding_config.required_dimension, 1024);
        assert_eq!(config.effective_db_path(), "/custom/path/db");
    }

    #[test]
    fn test_memex_config_with_embedding_config_syncs_dimension() {
        let embedding_config = EmbeddingConfig {
            required_dimension: 768,
            ..EmbeddingConfig::default()
        };

        let config = MemexConfig::new("sync-test", "ns").with_embedding_config(embedding_config);

        assert_eq!(config.dimension, 768);
        assert_eq!(config.embedding_config.required_dimension, 768);
    }

    #[test]
    fn test_memex_config_sync_dimension_fields_uses_non_default_embedding_dimension() {
        let mut config = MemexConfig::default();
        config.embedding_config.required_dimension = 1024;

        config.sync_dimension_fields().unwrap();

        assert_eq!(config.dimension, 1024);
        assert_eq!(config.embedding_config.required_dimension, 1024);
    }

    #[test]
    fn test_memex_config_sync_dimension_fields_rejects_true_conflict() {
        let mut config = MemexConfig {
            dimension: 768,
            ..MemexConfig::default()
        };
        config.embedding_config.required_dimension = 1024;

        let err = config.sync_dimension_fields().unwrap_err().to_string();
        assert!(err.contains("conflicts with embedding_config.required_dimension"));
    }

    #[test]
    fn test_store_item() {
        let item = StoreItem::new("doc-1", "Hello world")
            .with_metadata(serde_json::json!({"type": "greeting"}));

        assert_eq!(item.id, "doc-1");
        assert_eq!(item.text, "Hello world");
        assert_eq!(item.metadata["type"], "greeting");
    }

    #[test]
    fn test_store_item_default_metadata() {
        let item = StoreItem::new("doc-1", "Hello world");

        assert_eq!(item.id, "doc-1");
        assert_eq!(item.text, "Hello world");
        assert!(item.metadata.is_object());
        assert!(item.metadata.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_meta_filter_empty_matches_all() {
        let filter = MetaFilter::default();

        // Empty filter should match any metadata
        let any_metadata = serde_json::json!({
            "patient_id": "P-123",
            "visit_id": "V-456",
            "random_field": "value"
        });
        assert!(filter.matches(&any_metadata));

        // Even empty metadata should match
        let empty = serde_json::json!({});
        assert!(filter.matches(&empty));
    }

    #[test]
    fn test_meta_filter_date_range() {
        let filter = MetaFilter {
            date_from: Some("2024-01-01".to_string()),
            date_to: Some("2024-12-31".to_string()),
            ..Default::default()
        };

        // Within range
        let in_range = serde_json::json!({
            "date": "2024-06-15"
        });
        assert!(filter.matches(&in_range));

        // Before range
        let before = serde_json::json!({
            "date": "2023-12-31"
        });
        assert!(!filter.matches(&before));

        // After range
        let after = serde_json::json!({
            "date": "2025-01-01"
        });
        assert!(!filter.matches(&after));

        // No date field still matches (filter only applies if field exists)
        let no_date = serde_json::json!({
            "patient_id": "P-123"
        });
        assert!(filter.matches(&no_date));
    }

    #[test]
    fn test_meta_filter_for_visit() {
        let filter = MetaFilter::for_visit("V-789");

        let matching = serde_json::json!({
            "visit_id": "V-789",
            "patient_id": "P-123"
        });
        assert!(filter.matches(&matching));

        let not_matching = serde_json::json!({
            "visit_id": "V-other",
            "patient_id": "P-123"
        });
        assert!(!filter.matches(&not_matching));
    }

    #[test]
    fn test_meta_filter_combined() {
        let filter = MetaFilter {
            patient_id: Some("P-123".to_string()),
            doc_type: Some("soap_note".to_string()),
            ..Default::default()
        };

        // Both match
        let both_match = serde_json::json!({
            "patient_id": "P-123",
            "doc_type": "soap_note"
        });
        assert!(filter.matches(&both_match));

        // One doesn't match
        let wrong_type = serde_json::json!({
            "patient_id": "P-123",
            "doc_type": "prescription"
        });
        assert!(!filter.matches(&wrong_type));

        // Missing required field
        let missing = serde_json::json!({
            "patient_id": "P-123"
        });
        assert!(!filter.matches(&missing));
    }

    #[test]
    fn test_batch_result_struct() {
        let result = BatchResult {
            success_count: 10,
            failure_count: 2,
            failed_ids: vec!["doc-5".to_string(), "doc-8".to_string()],
        };

        assert_eq!(result.success_count, 10);
        assert_eq!(result.failure_count, 2);
        assert_eq!(result.failed_ids.len(), 2);
        assert!(result.failed_ids.contains(&"doc-5".to_string()));
    }

    #[test]
    fn test_memex_config_with_bm25() {
        use crate::search::BM25Config;

        let bm25_config = BM25Config::default();
        let config = MemexConfig::new("test-app", "docs").with_bm25(bm25_config);

        assert!(config.enable_bm25);
        assert!(config.bm25_config.is_some());
    }

    #[test]
    fn test_memex_config_effective_bm25_path() {
        let config = MemexConfig::new("my-app", "docs");
        assert_eq!(config.effective_bm25_path(), "~/.rmcp-servers/my-app/bm25");
    }

    #[test]
    fn test_resolved_bm25_config_uses_app_specific_path_for_hybrid_defaults() {
        let config = MemexConfig::new("my-app", "docs");
        let bm25 = config
            .resolved_bm25_config()
            .expect("hybrid defaults should provision BM25");

        assert_eq!(bm25.index_path, "~/.rmcp-servers/my-app/bm25");
    }

    #[test]
    fn test_resolved_hybrid_config_reuses_resolved_bm25_path() {
        let config = MemexConfig::new("my-app", "docs");
        let hybrid = config.resolved_hybrid_config();

        assert_eq!(hybrid.bm25.index_path, "~/.rmcp-servers/my-app/bm25");
    }

    #[test]
    fn test_meta_filter_serialization() {
        let filter = MetaFilter::for_patient("P-123").with_custom("status", "active");

        let json = serde_json::to_string(&filter).unwrap();
        let deserialized: MetaFilter = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.patient_id, Some("P-123".to_string()));
        assert_eq!(deserialized.custom.len(), 1);
        assert_eq!(
            deserialized.custom[0],
            ("status".to_string(), "active".to_string())
        );
    }

    #[test]
    fn test_memex_config_serialization() {
        let config = MemexConfig::new("test", "ns")
            .with_dimension(512)
            .with_db_path("/tmp/test");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MemexConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.app_name, "test");
        assert_eq!(deserialized.namespace, "ns");
        assert_eq!(deserialized.dimension, 512);
        assert_eq!(deserialized.embedding_config.required_dimension, 512);
        assert_eq!(deserialized.db_path, Some("/tmp/test".to_string()));
    }

    #[test]
    fn test_store_item_serialization() {
        let item =
            StoreItem::new("id-1", "content").with_metadata(serde_json::json!({"key": "value"}));

        let json = serde_json::to_string(&item).unwrap();
        let deserialized: StoreItem = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "id-1");
        assert_eq!(deserialized.text, "content");
        assert_eq!(deserialized.metadata["key"], "value");
    }
}
