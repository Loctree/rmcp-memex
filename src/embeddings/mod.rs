//! Universal embedding client with config-driven provider cascade.
//!
//! Supports any OpenAI-compatible embedding API (Ollama, vLLM, TEI, etc.)
//! Providers are tried in priority order until one responds.
//!
//! # Example config.toml
//! ```toml
//! [embeddings]
//! required_dimension = 4096
//! max_batch_chars = 32000
//! max_batch_items = 16
//!
//! [[embeddings.providers]]
//! name = "ollama-local"
//! base_url = "http://localhost:11434"
//! model = "qwen3-embedding:8b"
//! priority = 1
//!
//! [[embeddings.providers]]
//! name = "dragon"
//! base_url = "http://dragon:12345"
//! model = "Qwen/Qwen3-Embedding-4B"
//! priority = 2
//! ```

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// =============================================================================
// REQUEST/RESPONSE TYPES (OpenAI-compatible)
// =============================================================================

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    index: usize,
    score: f32,
}

// =============================================================================
// PROVIDER CONFIGURATION
// =============================================================================

/// Single embedding provider configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProviderConfig {
    /// Human-readable name for logging
    #[serde(default)]
    pub name: String,
    /// Base URL (e.g., "http://localhost:11434")
    #[serde(default)]
    pub base_url: String,
    /// Model name to use
    #[serde(default)]
    pub model: String,
    /// Priority (1 = highest, tried first)
    #[serde(default = "default_priority")]
    pub priority: u8,
    /// Embedding endpoint path (default: /v1/embeddings)
    #[serde(default = "default_embeddings_endpoint")]
    pub endpoint: String,
}

fn default_priority() -> u8 {
    10
}

fn default_embeddings_endpoint() -> String {
    "/v1/embeddings".to_string()
}

/// Reranker configuration (optional, separate from embedders)
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RerankerConfig {
    /// Base URL for reranker service
    pub base_url: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Endpoint path (default: /v1/rerank)
    #[serde(default = "default_rerank_endpoint")]
    pub endpoint: String,
}

fn default_rerank_endpoint() -> String {
    "/v1/rerank".to_string()
}

fn default_dimension() -> usize {
    4096
}

fn default_max_batch_chars() -> usize {
    128000 // Increased 4x for better GPU utilization
}

fn default_max_batch_items() -> usize {
    64 // Increased 4x - fewer API calls, better throughput
}

/// Complete embedding configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    /// Required vector dimension (mismatch corrupts database!)
    #[serde(default = "default_dimension")]
    pub required_dimension: usize,
    /// Maximum characters per embedding batch to avoid OOM (default: 32000)
    #[serde(default = "default_max_batch_chars")]
    pub max_batch_chars: usize,
    /// Maximum items per embedding batch (default: 16)
    #[serde(default = "default_max_batch_items")]
    pub max_batch_items: usize,
    /// List of providers to try in priority order
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Optional reranker configuration
    #[serde(default)]
    pub reranker: RerankerConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            required_dimension: 4096,
            max_batch_chars: default_max_batch_chars(),
            max_batch_items: default_max_batch_items(),
            providers: vec![
                ProviderConfig {
                    name: "ollama-local".to_string(),
                    base_url: "http://localhost:11434".to_string(),
                    model: "qwen3-embedding:8b".to_string(),
                    priority: 1,
                    endpoint: default_embeddings_endpoint(),
                },
                ProviderConfig {
                    name: "dragon".to_string(),
                    base_url: "http://dragon:12345".to_string(),
                    model: "Qwen/Qwen3-Embedding-4B".to_string(),
                    priority: 2,
                    endpoint: default_embeddings_endpoint(),
                },
            ],
            reranker: RerankerConfig::default(),
        }
    }
}

impl EmbeddingConfig {
    /// Returns the name of the first (highest priority) provider
    pub fn provider_name(&self) -> String {
        self.providers
            .first()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "none".to_string())
    }

    /// Returns the model name of the first (highest priority) provider
    pub fn model_name(&self) -> String {
        self.providers
            .first()
            .map(|p| p.model.clone())
            .unwrap_or_else(|| "none".to_string())
    }

    /// Alias for required_dimension for API compatibility
    pub fn dimension(&self) -> usize {
        self.required_dimension
    }
}

// =============================================================================
// LEGACY CONFIG (backward compatibility)
// =============================================================================

/// Legacy MLX configuration - deprecated, use EmbeddingConfig instead
#[derive(Debug, Clone)]
pub struct MlxConfig {
    pub disabled: bool,
    pub local_port: u16,
    pub dragon_url: String,
    pub dragon_port: u16,
    pub embedder_model: String,
    pub reranker_model: String,
    pub reranker_port_offset: u16,
    pub max_batch_chars: usize,
    pub max_batch_items: usize,
}

/// Options for merging file config into MlxConfig
#[derive(Debug, Clone, Default)]
pub struct MlxMergeOptions {
    pub disabled: Option<bool>,
    pub local_port: Option<u16>,
    pub dragon_url: Option<String>,
    pub dragon_port: Option<u16>,
    pub embedder_model: Option<String>,
    pub reranker_model: Option<String>,
    pub reranker_port_offset: Option<u16>,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            local_port: 12345,
            dragon_url: "http://dragon".to_string(),
            dragon_port: 12345,
            embedder_model: "Qwen/Qwen3-Embedding-4B".to_string(),
            reranker_model: "Qwen/Qwen3-Reranker-4B".to_string(),
            reranker_port_offset: 1,
            max_batch_chars: default_max_batch_chars(),
            max_batch_items: default_max_batch_items(),
        }
    }
}

impl MlxConfig {
    /// Create config from environment variables (legacy support)
    pub fn from_env() -> Self {
        let disabled = std::env::var("DISABLE_MLX")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let local_port = std::env::var("EMBEDDER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12345);

        let dragon_url =
            std::env::var("DRAGON_BASE_URL").unwrap_or_else(|_| "http://dragon".to_string());

        let dragon_port = std::env::var("DRAGON_EMBEDDER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(local_port);

        let reranker_port_offset = std::env::var("RERANKER_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .map(|rp| rp.saturating_sub(local_port))
            .unwrap_or(1);

        let embedder_model = std::env::var("EMBEDDER_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-Embedding-4B".to_string());

        let reranker_model = std::env::var("RERANKER_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-Reranker-4B".to_string());

        let max_batch_chars = std::env::var("MLX_MAX_BATCH_CHARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32000);

        let max_batch_items = std::env::var("MLX_MAX_BATCH_ITEMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        Self {
            disabled,
            local_port,
            dragon_url,
            dragon_port,
            embedder_model,
            reranker_model,
            reranker_port_offset,
            max_batch_chars,
            max_batch_items,
        }
    }

    /// Merge with values from file config
    pub fn merge_file_config(&mut self, opts: MlxMergeOptions) {
        if let Some(v) = opts.disabled {
            self.disabled = v;
        }
        if let Some(v) = opts.local_port {
            self.local_port = v;
        }
        if let Some(v) = opts.dragon_url {
            self.dragon_url = v;
        }
        if let Some(v) = opts.dragon_port {
            self.dragon_port = v;
        }
        if let Some(v) = opts.embedder_model {
            self.embedder_model = v;
        }
        if let Some(v) = opts.reranker_model {
            self.reranker_model = v;
        }
        if let Some(v) = opts.reranker_port_offset {
            self.reranker_port_offset = v;
        }
    }

    /// Convert legacy config to new EmbeddingConfig
    pub fn to_embedding_config(&self) -> EmbeddingConfig {
        let reranker_port = self.local_port + self.reranker_port_offset;

        EmbeddingConfig {
            required_dimension: 4096,
            max_batch_chars: self.max_batch_chars,
            max_batch_items: self.max_batch_items,
            providers: vec![
                ProviderConfig {
                    name: "local".to_string(),
                    base_url: format!("http://localhost:{}", self.local_port),
                    model: self.embedder_model.clone(),
                    priority: 1,
                    endpoint: default_embeddings_endpoint(),
                },
                ProviderConfig {
                    name: "dragon".to_string(),
                    base_url: format!("{}:{}", self.dragon_url, self.dragon_port),
                    model: self.embedder_model.clone(),
                    priority: 2,
                    endpoint: default_embeddings_endpoint(),
                },
            ],
            reranker: RerankerConfig {
                base_url: Some(format!("{}:{}", self.dragon_url, reranker_port)),
                model: Some(self.reranker_model.clone()),
                endpoint: default_rerank_endpoint(),
            },
        }
    }

    /// Set batch limits
    pub fn with_batch_limits(mut self, max_chars: usize, max_items: usize) -> Self {
        self.max_batch_chars = max_chars;
        self.max_batch_items = max_items;
        self
    }
}

// =============================================================================
// EMBEDDING CLIENT
// =============================================================================

/// Universal embedding client with provider cascade
pub struct EmbeddingClient {
    client: Client,
    embedder_url: String,
    embedder_model: String,
    reranker_url: Option<String>,
    reranker_model: Option<String>,
    /// Which provider we're connected to
    connected_to: String,
    /// Expected dimension (for validation)
    required_dimension: usize,
    /// Maximum characters per embedding batch
    max_batch_chars: usize,
    /// Maximum items per embedding batch
    max_batch_items: usize,
}

// Type alias for backward compatibility
pub type MLXBridge = EmbeddingClient;

impl EmbeddingClient {
    /// Create client with config-driven provider cascade
    pub async fn new(config: &EmbeddingConfig) -> Result<Self> {
        if config.providers.is_empty() {
            return Err(anyhow!(
                "No embedding providers configured! Add providers to [embeddings.providers]"
            ));
        }

        // Long timeout for large embedding batches (100+ chunks can take minutes)
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        // Sort providers by priority
        let mut providers = config.providers.clone();
        providers.sort_by_key(|p| p.priority);

        // Try each provider in order
        let mut tried = Vec::new();
        for provider in &providers {
            let base_url = provider.base_url.trim_end_matches('/');

            match Self::health_check(&client, base_url).await {
                Ok(()) => {
                    tracing::info!("Embedding: Connected to {} ({})", provider.name, base_url);

                    let embedder_url = format!("{}{}", base_url, provider.endpoint);

                    // Build reranker URL if configured
                    let (reranker_url, reranker_model) =
                        if let Some(ref rr_base) = config.reranker.base_url {
                            (
                                Some(format!(
                                    "{}{}",
                                    rr_base.trim_end_matches('/'),
                                    config.reranker.endpoint
                                )),
                                config.reranker.model.clone(),
                            )
                        } else {
                            (None, None)
                        };

                    // FAIL-FAST: Test embedding dimension before accepting this provider
                    let test_dim = Self::test_dimension(
                        &client,
                        &embedder_url,
                        &provider.model,
                        config.required_dimension,
                    )
                    .await;

                    match test_dim {
                        Ok(actual_dim) => {
                            tracing::info!(
                                "Embedding: Dimension verified: {} (required: {})",
                                actual_dim,
                                config.required_dimension
                            );
                            return Ok(Self {
                                client,
                                embedder_url,
                                embedder_model: provider.model.clone(),
                                reranker_url,
                                reranker_model,
                                connected_to: provider.name.clone(),
                                required_dimension: config.required_dimension,
                                max_batch_chars: config.max_batch_chars,
                                max_batch_items: config.max_batch_items,
                            });
                        }
                        Err(e) => {
                            tracing::error!(
                                "Embedding: {} dimension check FAILED: {}",
                                provider.name,
                                e
                            );
                            tried.push(format!(
                                "- {} ({}): dimension check failed: {}",
                                provider.name, base_url, e
                            ));
                            // Continue to next provider
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Embedding: {} ({}) unavailable: {}",
                        provider.name,
                        base_url,
                        e
                    );
                    tried.push(format!("- {} ({}): {}", provider.name, base_url, e));
                }
            }
        }

        // All providers failed
        Err(anyhow!(
            "All embedding providers unavailable!\nTried:\n{}",
            tried.join("\n")
        ))
    }

    /// Create from legacy MlxConfig (backward compatibility)
    pub async fn from_legacy(config: &MlxConfig) -> Result<Self> {
        if config.disabled {
            return Err(anyhow!(
                "Embedding disabled via config. No fallback available!"
            ));
        }
        tracing::warn!("Using legacy [mlx] config - please migrate to [embeddings.providers]");
        let embedding_config = config.to_embedding_config();
        Self::new(&embedding_config).await
    }

    /// Legacy constructor from env vars only
    pub async fn from_env() -> Result<Self> {
        let config = MlxConfig::from_env();
        Self::from_legacy(&config).await
    }

    async fn health_check(client: &Client, base_url: &str) -> Result<()> {
        // Try /v1/models first (OpenAI-compat)
        let url = format!("{}/v1/models", base_url);
        let response = client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) if resp.status().as_u16() == 404 => {
                // Try Ollama-native endpoint
                let ollama_url = format!("{}/api/tags", base_url);
                let ollama_resp = client
                    .get(&ollama_url)
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await?;
                if ollama_resp.status().is_success() {
                    Ok(())
                } else {
                    Err(anyhow!("Neither /v1/models nor /api/tags available"))
                }
            }
            Ok(resp) => Err(anyhow!("Health check failed: {}", resp.status())),
            Err(e) => Err(anyhow!("Connection failed: {}", e)),
        }
    }

    /// Test embedding dimension by sending a probe request.
    /// Returns actual dimension if it matches required, otherwise error.
    async fn test_dimension(
        client: &Client,
        embedder_url: &str,
        model: &str,
        required_dimension: usize,
    ) -> Result<usize> {
        let request = EmbeddingRequest {
            input: vec!["dimension test".to_string()],
            model: model.to_string(),
        };

        let response = client
            .post(embedder_url)
            .json(&request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| anyhow!("Dimension probe failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Dimension probe returned {}: {}",
                status,
                body.chars().take(200).collect::<String>()
            ));
        }

        let embed_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse dimension probe response: {}", e))?;

        let actual_dim = embed_response
            .data
            .first()
            .map(|d| d.embedding.len())
            .ok_or_else(|| anyhow!("No embedding in dimension probe response"))?;

        if actual_dim != required_dimension {
            return Err(anyhow!(
                "DIMENSION MISMATCH: model returns {} dims, but database requires {}. \
                 Using this provider would CORRUPT the database!",
                actual_dim,
                required_dimension
            ));
        }

        Ok(actual_dim)
    }

    /// Get which provider we're connected to
    pub fn connected_to(&self) -> &str {
        &self.connected_to
    }

    /// Get required dimension
    pub fn required_dimension(&self) -> usize {
        self.required_dimension
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let text_preview: String = text.chars().take(100).collect();
        tracing::debug!(
            "Embedding single text ({} chars): {}{}",
            text.chars().count(),
            text_preview,
            if text.chars().count() > 100 {
                "..."
            } else {
                ""
            }
        );

        let request = EmbeddingRequest {
            input: vec![text.to_string()],
            model: self.embedder_model.clone(),
        };

        let response = match self
            .client
            .post(&self.embedder_url)
            .json(&request)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!(
                    "Embedding request failed: {:?}\n  URL: {}\n  Model: {}",
                    e,
                    self.embedder_url,
                    self.embedder_model
                );
                return Err(anyhow!("Embedding request failed: {}", e));
            }
        };

        let status = response.status();
        let response_text = response.text().await.unwrap_or_else(|e| {
            tracing::warn!("Failed to read response body: {:?}", e);
            "<failed to read body>".to_string()
        });

        if !status.is_success() {
            tracing::error!(
                "Embedding API error (HTTP {}):\n  URL: {}\n  Model: {}\n  Response: {}",
                status,
                self.embedder_url,
                self.embedder_model,
                response_text
            );
            return Err(anyhow!(
                "Embedding API error (HTTP {}): {}",
                status,
                response_text
            ));
        }

        let parsed: EmbeddingResponse = match serde_json::from_str(&response_text) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "Failed to parse embedding response: {:?}\n  Response body: {}",
                    e,
                    response_text
                );
                return Err(anyhow!("Failed to parse embedding response: {}", e));
            }
        };

        let embedding = parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| {
                tracing::error!("No embedding returned in response: {}", response_text);
                anyhow!("No embedding returned")
            })?;

        // Validate dimension
        if embedding.len() != self.required_dimension {
            tracing::error!(
                "Dimension mismatch! Expected {}, got {}. Model: {}",
                self.required_dimension,
                embedding.len(),
                self.embedder_model
            );
            return Err(anyhow!(
                "Dimension mismatch! Expected {}, got {}. This would corrupt the database!",
                self.required_dimension,
                embedding.len()
            ));
        }

        tracing::debug!("Successfully embedded text ({} dims)", embedding.len());
        Ok(embedding)
    }

    /// Embed a batch of texts with intelligent batching to avoid OOM.
    ///
    /// Large texts are chunked and only the first chunk is embedded.
    /// Batches are split to stay under max_batch_chars and max_batch_items.
    /// Failed chunks are retried individually with exponential backoff.
    pub async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());
        let mut current_batch: Vec<String> = Vec::new();
        let mut current_batch_indices: Vec<usize> = Vec::new();
        let mut current_chars = 0;

        // Max chars per individual text (half of batch limit for safety)
        let max_text_chars = self.max_batch_chars / 2;

        // Prepare all texts first
        let prepared_texts: Vec<String> = texts
            .iter()
            .map(|text| {
                let char_count = text.chars().count();
                if char_count > max_text_chars {
                    tracing::debug!(
                        "Text too large ({} chars), truncating to {} chars",
                        char_count,
                        max_text_chars
                    );
                    truncate_at_boundary(text, max_text_chars)
                } else {
                    text.clone()
                }
            })
            .collect();

        // Pre-allocate result vector with None
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut failed_indices: Vec<usize> = Vec::new();

        for (idx, text_to_embed) in prepared_texts.iter().enumerate() {
            let text_len = text_to_embed.chars().count();

            // Check if we need to flush current batch
            if !current_batch.is_empty()
                && (current_chars + text_len > self.max_batch_chars
                    || current_batch.len() >= self.max_batch_items)
            {
                // Flush current batch with retry
                match self.embed_batch_internal(&current_batch).await {
                    Ok(batch_embeddings) => {
                        for (i, emb) in batch_embeddings.into_iter().enumerate() {
                            if let Some(orig_idx) = current_batch_indices.get(i) {
                                results[*orig_idx] = Some(emb);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Batch embedding failed for {} texts, will retry individually: {}",
                            current_batch.len(),
                            e
                        );
                        failed_indices.extend(current_batch_indices.iter().copied());
                    }
                }
                current_batch.clear();
                current_batch_indices.clear();
                current_chars = 0;
            }

            current_batch.push(text_to_embed.clone());
            current_batch_indices.push(idx);
            current_chars += text_len;
        }

        // Flush remaining batch
        if !current_batch.is_empty() {
            match self.embed_batch_internal(&current_batch).await {
                Ok(batch_embeddings) => {
                    for (i, emb) in batch_embeddings.into_iter().enumerate() {
                        if let Some(orig_idx) = current_batch_indices.get(i) {
                            results[*orig_idx] = Some(emb);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Batch embedding failed for {} texts, will retry individually: {}",
                        current_batch.len(),
                        e
                    );
                    failed_indices.extend(current_batch_indices.iter().copied());
                }
            }
        }

        // Retry failed chunks individually with exponential backoff
        const MAX_RETRIES: usize = 3;
        for idx in failed_indices {
            let text = &prepared_texts[idx];
            let mut attempts = 0;
            let mut last_error = String::new();

            while attempts < MAX_RETRIES {
                match self.embed(text).await {
                    Ok(embedding) => {
                        results[idx] = Some(embedding);
                        tracing::info!(
                            "Retry succeeded for chunk {} after {} attempts",
                            idx,
                            attempts + 1
                        );
                        break;
                    }
                    Err(e) => {
                        attempts += 1;
                        last_error = e.to_string();
                        tracing::warn!(
                            "Embed attempt {}/{} failed for chunk {}: {}",
                            attempts,
                            MAX_RETRIES,
                            idx,
                            e
                        );
                        if attempts < MAX_RETRIES {
                            // Exponential backoff: 100ms, 200ms, 400ms
                            let delay_ms = 100 * (1 << attempts);
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        }
                    }
                }
            }

            if results[idx].is_none() {
                tracing::error!(
                    "Chunk {} failed after {} retries: {}",
                    idx,
                    MAX_RETRIES,
                    last_error
                );
                return Err(anyhow!(
                    "Failed to embed chunk {} after {} retries: {}",
                    idx,
                    MAX_RETRIES,
                    last_error
                ));
            }
        }

        // Collect all results - all should be Some at this point
        for (idx, opt) in results.iter().enumerate() {
            match opt {
                Some(emb) => all_embeddings.push(emb.clone()),
                None => {
                    return Err(anyhow!(
                        "Internal error: missing embedding for chunk {}",
                        idx
                    ));
                }
            }
        }

        Ok(all_embeddings)
    }

    /// Internal batch embedding - sends directly to server
    async fn embed_batch_internal(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let total_chars: usize = texts.iter().map(|t| t.chars().count()).sum();

        tracing::debug!(
            "Embedding batch: {} texts, {} chars total",
            texts.len(),
            total_chars
        );

        // Log first few chars of each text in trace mode for debugging
        for (i, text) in texts.iter().enumerate() {
            let preview: String = text.chars().take(50).collect();
            tracing::trace!(
                "  Batch[{}]: {} chars - {}{}",
                i,
                text.chars().count(),
                preview,
                if text.chars().count() > 50 { "..." } else { "" }
            );
        }

        let request = EmbeddingRequest {
            input: texts.to_vec(),
            model: self.embedder_model.clone(),
        };

        // Retry with exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (max)
        const MAX_BATCH_RETRIES: usize = 10;
        const MAX_BACKOFF_SECS: u64 = 30;
        let mut attempt = 0;

        loop {
            attempt += 1;
            let response = match self
                .client
                .post(&self.embedder_url)
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    if attempt >= MAX_BATCH_RETRIES {
                        tracing::error!(
                            "Batch embedding failed after {} retries: {:?}\n  URL: {}\n  Model: {}",
                            MAX_BATCH_RETRIES,
                            e,
                            self.embedder_url,
                            self.embedder_model
                        );
                        return Err(anyhow!(
                            "Embedding request failed after {} retries: {}",
                            MAX_BATCH_RETRIES,
                            e
                        ));
                    }

                    // Exponential backoff with cap
                    let backoff_secs = (1u64 << attempt.min(5)).min(MAX_BACKOFF_SECS);
                    tracing::warn!(
                        "Embedding request failed (attempt {}/{}), retrying in {}s: {}",
                        attempt,
                        MAX_BATCH_RETRIES,
                        backoff_secs,
                        e
                    );
                    tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                    continue;
                }
            };

            // Success - process response
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();

                if attempt >= MAX_BATCH_RETRIES {
                    tracing::error!(
                        "Embedding API error after {} retries: {} - {}",
                        MAX_BATCH_RETRIES,
                        status,
                        body
                    );
                    return Err(anyhow!("Embedding API error: {} - {}", status, body));
                }

                let backoff_secs = (1u64 << attempt.min(5)).min(MAX_BACKOFF_SECS);
                tracing::warn!(
                    "Embedding API error (attempt {}/{}), retrying in {}s: {} - {}",
                    attempt,
                    MAX_BATCH_RETRIES,
                    backoff_secs,
                    status,
                    body
                );
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                continue;
            }

            // Parse response
            let embedding_response: EmbeddingResponse = match response.json().await {
                Ok(r) => r,
                Err(e) => {
                    if attempt >= MAX_BATCH_RETRIES {
                        return Err(anyhow!("Failed to parse embedding response: {}", e));
                    }
                    let backoff_secs = (1u64 << attempt.min(5)).min(MAX_BACKOFF_SECS);
                    tracing::warn!(
                        "Failed to parse response (attempt {}/{}), retrying in {}s: {}",
                        attempt,
                        MAX_BATCH_RETRIES,
                        backoff_secs,
                        e
                    );
                    tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                    continue;
                }
            };

            // Validate dimensions
            let embeddings: Vec<Vec<f32>> = embedding_response
                .data
                .into_iter()
                .map(|d| d.embedding)
                .collect();

            if embeddings.len() != texts.len() {
                return Err(anyhow!(
                    "Embedding count mismatch: got {} embeddings for {} texts",
                    embeddings.len(),
                    texts.len()
                ));
            }

            if let Some(first) = embeddings.first()
                && first.len() != self.required_dimension
            {
                return Err(anyhow!(
                    "Dimension mismatch: expected {}, got {}",
                    self.required_dimension,
                    first.len()
                ));
            }

            return Ok(embeddings);
        }
    }

    pub async fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>> {
        let reranker_url = self.reranker_url.as_ref().ok_or_else(|| {
            anyhow!("Reranker not configured. Add [embeddings.reranker] to config.")
        })?;
        let reranker_model = self
            .reranker_model
            .as_ref()
            .ok_or_else(|| anyhow!("Reranker model not configured."))?;

        let query_preview: String = query.chars().take(100).collect();
        tracing::debug!(
            "Reranking {} documents for query: {}{}",
            documents.len(),
            query_preview,
            if query.chars().count() > 100 {
                "..."
            } else {
                ""
            }
        );

        let request = RerankRequest {
            query: query.to_string(),
            documents: documents.to_vec(),
            model: reranker_model.clone(),
        };

        let response = match self.client.post(reranker_url).json(&request).send().await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!(
                    "Rerank request failed: {:?}\n  URL: {}\n  Model: {}\n  Query: {}\n  Documents: {}",
                    e,
                    reranker_url,
                    reranker_model,
                    query_preview,
                    documents.len()
                );
                return Err(anyhow!("Rerank request failed: {}", e));
            }
        };

        let status = response.status();
        let response_text = response.text().await.unwrap_or_else(|e| {
            tracing::warn!("Failed to read rerank response body: {:?}", e);
            "<failed to read body>".to_string()
        });

        if !status.is_success() {
            tracing::error!(
                "Rerank API error (HTTP {}):\n  URL: {}\n  Model: {}\n  Response: {}",
                status,
                reranker_url,
                reranker_model,
                response_text
            );
            return Err(anyhow!(
                "Rerank API error (HTTP {}): {}",
                status,
                response_text
            ));
        }

        let parsed: RerankResponse = match serde_json::from_str(&response_text) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "Failed to parse rerank response: {:?}\n  Response body: {}",
                    e,
                    response_text
                );
                return Err(anyhow!("Failed to parse rerank response: {}", e));
            }
        };

        tracing::debug!("Rerank complete: {} documents scored", parsed.results.len());

        Ok(parsed
            .results
            .into_iter()
            .map(|r| (r.index, r.score))
            .collect())
    }
}

/// Truncate text at a word/sentence boundary to avoid cutting mid-word (UTF-8 safe)
fn truncate_at_boundary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    // Get byte index of max_chars-th character (UTF-8 safe)
    let byte_idx = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());

    let truncated = &text[..byte_idx];

    // Try to find a sentence boundary first (prefer complete sentences)
    let half_byte_idx = text
        .char_indices()
        .nth(max_chars / 2)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    if let Some(pos) = truncated.rfind(['.', '!', '?', '\n'])
        && pos > half_byte_idx
    {
        return text[..=pos].to_string();
    }

    // Fall back to word boundary
    if let Some(pos) = truncated.rfind([' ', '\t', '\n']) {
        return text[..pos].to_string();
    }

    // Last resort: hard truncate
    truncated.to_string()
}

// =============================================================================
// TOKEN-AWARE VALIDATION
// =============================================================================
//
// Embedding models have token limits (e.g., 8192 for qwen3-embedding).
// These utilities estimate token counts and validate chunks before embedding.
// =============================================================================

/// Token estimation configuration
#[derive(Debug, Clone)]
pub struct TokenConfig {
    /// Maximum tokens for the embedding model
    pub max_tokens: usize,
    /// Average characters per token (varies by language)
    /// English: ~4 chars/token, Polish/multilingual: ~2-3 chars/token
    pub chars_per_token: f32,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            max_tokens: 8192,     // qwen3-embedding default
            chars_per_token: 3.0, // Conservative for multilingual
        }
    }
}

impl TokenConfig {
    /// Create config for English-only content
    pub fn english() -> Self {
        Self {
            max_tokens: 8192,
            chars_per_token: 4.0,
        }
    }

    /// Create config for multilingual/Polish content
    pub fn multilingual() -> Self {
        Self {
            max_tokens: 8192,
            chars_per_token: 2.5,
        }
    }

    /// Create config with custom max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

/// Estimate token count for text
///
/// This is a heuristic approximation. For precise counting,
/// use the actual tokenizer (tiktoken, sentencepiece, etc.)
pub fn estimate_tokens(text: &str, config: &TokenConfig) -> usize {
    let char_count = text.chars().count();
    (char_count as f32 / config.chars_per_token).ceil() as usize
}

/// Validate that a chunk fits within token limits
///
/// Returns Ok(()) if chunk is within limits, Err with details otherwise.
pub fn validate_chunk_tokens(chunk: &str, config: &TokenConfig) -> Result<()> {
    let estimated = estimate_tokens(chunk, config);

    if estimated > config.max_tokens {
        return Err(anyhow!(
            "Chunk exceeds token limit: ~{} tokens > {} max (text: {} chars). \
             Consider reducing chunk_size or enabling truncation.",
            estimated,
            config.max_tokens,
            chunk.chars().count()
        ));
    }

    Ok(())
}

/// Calculate safe chunk size in characters for given token limit
pub fn safe_chunk_size(config: &TokenConfig) -> usize {
    // Use 80% of max to leave room for context prefix
    let safe_tokens = (config.max_tokens as f32 * 0.8) as usize;
    (safe_tokens as f32 * config.chars_per_token) as usize
}

/// Truncate text to fit within token limit
pub fn truncate_to_token_limit(text: &str, config: &TokenConfig) -> String {
    let safe_chars = safe_chunk_size(config);

    if text.chars().count() <= safe_chars {
        return text.to_string();
    }

    truncate_at_boundary(text, safe_chars)
}

/// Validate a batch of texts and return which ones exceed limits
pub fn validate_batch_tokens(texts: &[String], config: &TokenConfig) -> Vec<(usize, usize)> {
    texts
        .iter()
        .enumerate()
        .filter_map(|(idx, text)| {
            let estimated = estimate_tokens(text, config);
            if estimated > config.max_tokens {
                Some((idx, estimated))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_sorting() {
        let mut providers = [
            ProviderConfig {
                name: "low".into(),
                base_url: "http://a".into(),
                model: "m".into(),
                priority: 10,
                endpoint: "/v1/embeddings".into(),
            },
            ProviderConfig {
                name: "high".into(),
                base_url: "http://b".into(),
                model: "m".into(),
                priority: 1,
                endpoint: "/v1/embeddings".into(),
            },
        ];
        providers.sort_by_key(|p| p.priority);
        assert_eq!(providers[0].name, "high");
        assert_eq!(providers[1].name, "low");
    }

    #[test]
    fn test_legacy_conversion() {
        let legacy = MlxConfig {
            disabled: false,
            local_port: 12345,
            dragon_url: "http://dragon".into(),
            dragon_port: 12345,
            embedder_model: "test-model".into(),
            reranker_model: "rerank-model".into(),
            reranker_port_offset: 1,
            max_batch_chars: 32000,
            max_batch_items: 16,
        };
        let config = legacy.to_embedding_config();
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].base_url, "http://localhost:12345");
        assert!(config.reranker.base_url.is_some());
        assert_eq!(config.max_batch_chars, 32000);
        assert_eq!(config.max_batch_items, 16);
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.required_dimension, 4096);
        assert_eq!(config.max_batch_chars, 128000); // 4x larger for GPU efficiency
        assert_eq!(config.max_batch_items, 64); // 4x more items per batch
        assert!(!config.providers.is_empty());
    }

    #[test]
    fn test_truncate_at_boundary() {
        // Test sentence boundary
        let text = "Hello world. This is a test.";
        let truncated = truncate_at_boundary(text, 15);
        assert_eq!(truncated, "Hello world.");

        // Test word boundary fallback
        let text = "Hello world this is a test";
        let truncated = truncate_at_boundary(text, 15);
        assert_eq!(truncated, "Hello world");

        // Test no truncation needed
        let text = "Short text";
        let truncated = truncate_at_boundary(text, 100);
        assert_eq!(truncated, "Short text");
    }

    #[test]
    fn test_token_estimation() {
        let config = TokenConfig::default();

        // ~3 chars per token (default multilingual)
        let text = "Hello world"; // 11 chars -> ~4 tokens
        let tokens = estimate_tokens(text, &config);
        assert!((3..=5).contains(&tokens));

        // English config (4 chars per token)
        let english_config = TokenConfig::english();
        let tokens = estimate_tokens(text, &english_config);
        assert!((2..=4).contains(&tokens));
    }

    #[test]
    fn test_chunk_validation() {
        let config = TokenConfig::default().with_max_tokens(100);

        // Short text should pass
        let short = "Hello world";
        assert!(validate_chunk_tokens(short, &config).is_ok());

        // Long text should fail
        let long = "a".repeat(1000); // Way more than 100 * 3 = 300 chars
        assert!(validate_chunk_tokens(&long, &config).is_err());
    }

    #[test]
    fn test_safe_chunk_size() {
        let config = TokenConfig::default(); // 8192 tokens, 3 chars/token

        let safe = safe_chunk_size(&config);
        // 8192 * 0.8 * 3 = 19660 chars
        assert!(safe > 15000 && safe < 25000);
    }

    #[test]
    fn test_batch_validation() {
        let config = TokenConfig::default().with_max_tokens(10);

        let texts = vec![
            "short".to_string(),      // OK
            "a".repeat(100),          // Too long
            "also short".to_string(), // OK
            "b".repeat(200),          // Too long
        ];

        let failures = validate_batch_tokens(&texts, &config);
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].0, 1); // Index 1
        assert_eq!(failures[1].0, 3); // Index 3
    }
}

// =============================================================================
// DIMENSION ADAPTER - Cross-dimension embedding compatibility
// =============================================================================

/// Adapter for cross-dimension embedding compatibility.
///
/// Enables searching across databases with different embedding dimensions
/// (e.g., 1024, 2048, 4096) by expanding or contracting embeddings.
///
/// # Strategies
/// - **Expand**: Zero-pad smaller embeddings to target dimension
/// - **Contract**: Truncate or project larger embeddings to target dimension
///
/// # Example
/// ```rust,ignore
/// let adapter = DimensionAdapter::new(1024, 4096);
/// let expanded = adapter.expand(small_embedding);  // 1024 -> 4096
///
/// let adapter = DimensionAdapter::new(4096, 1024);
/// let contracted = adapter.contract(large_embedding);  // 4096 -> 1024
/// ```
#[derive(Debug, Clone)]
pub struct DimensionAdapter {
    /// Source embedding dimension
    pub source_dim: usize,
    /// Target embedding dimension
    pub target_dim: usize,
}

impl DimensionAdapter {
    /// Create a new dimension adapter
    pub fn new(source_dim: usize, target_dim: usize) -> Self {
        Self {
            source_dim,
            target_dim,
        }
    }

    /// Check if adaptation is needed
    pub fn needs_adaptation(&self) -> bool {
        self.source_dim != self.target_dim
    }

    /// Adapt embedding to target dimension (auto-detect expand/contract)
    pub fn adapt(&self, embedding: Vec<f32>) -> Vec<f32> {
        if embedding.len() == self.target_dim {
            return embedding;
        }

        if embedding.len() < self.target_dim {
            self.expand(embedding)
        } else {
            self.contract(embedding)
        }
    }

    /// Expand smaller embeddings to target dimension via zero-padding.
    ///
    /// Uses normalized zero-padding to minimize impact on cosine similarity.
    pub fn expand(&self, embedding: Vec<f32>) -> Vec<f32> {
        if embedding.len() >= self.target_dim {
            return embedding[..self.target_dim].to_vec();
        }

        let mut padded = embedding;
        padded.resize(self.target_dim, 0.0);

        // Re-normalize to unit length for cosine similarity
        self.normalize(&mut padded);
        padded
    }

    /// Contract larger embeddings to target dimension.
    ///
    /// Uses PCA-like projection for dimensions that are powers of 2,
    /// otherwise falls back to truncation.
    pub fn contract(&self, embedding: Vec<f32>) -> Vec<f32> {
        if embedding.len() <= self.target_dim {
            return embedding;
        }

        // For power-of-2 reductions (4096->2048, 2048->1024), use averaging
        // This preserves more information than truncation
        if self.is_power_of_two_reduction(embedding.len()) {
            self.average_reduction(embedding)
        } else {
            // Fallback to truncation
            embedding[..self.target_dim].to_vec()
        }
    }

    /// Check if this is a clean power-of-2 reduction (e.g., 4096->2048)
    fn is_power_of_two_reduction(&self, source_len: usize) -> bool {
        source_len > self.target_dim
            && source_len.is_power_of_two()
            && self.target_dim.is_power_of_two()
            && source_len.is_multiple_of(self.target_dim)
    }

    /// Reduce by averaging consecutive elements (preserves information better than truncation)
    fn average_reduction(&self, embedding: Vec<f32>) -> Vec<f32> {
        let factor = embedding.len() / self.target_dim;
        let mut result = Vec::with_capacity(self.target_dim);

        for chunk in embedding.chunks(factor) {
            let sum: f32 = chunk.iter().sum();
            result.push(sum / factor as f32);
        }

        // Re-normalize
        self.normalize(&mut result);
        result
    }

    /// Normalize vector to unit length (L2 norm)
    fn normalize(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }
}

/// Perform cross-dimension search by adapting query embedding
pub fn cross_dimension_search_adapt(query_embedding: Vec<f32>, target_dim: usize) -> Vec<f32> {
    let adapter = DimensionAdapter::new(query_embedding.len(), target_dim);
    adapter.adapt(query_embedding)
}

#[cfg(test)]
mod dimension_adapter_tests {
    use super::*;

    #[test]
    fn test_expand_1024_to_4096() {
        let adapter = DimensionAdapter::new(1024, 4096);
        let small = vec![0.1f32; 1024];
        let expanded = adapter.expand(small);

        assert_eq!(expanded.len(), 4096);
        // First 1024 should be non-zero (after normalization)
        assert!(expanded[0].abs() > 1e-10);
        // Last elements should be zero
        assert!(expanded[4095].abs() < 1e-10);
    }

    #[test]
    fn test_contract_4096_to_1024() {
        let adapter = DimensionAdapter::new(4096, 1024);
        let large = vec![0.1f32; 4096];
        let contracted = adapter.contract(large);

        assert_eq!(contracted.len(), 1024);
        // Should be normalized
        let norm: f32 = contracted.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_auto_detect() {
        let adapter = DimensionAdapter::new(1024, 4096);

        // Small to large (expand)
        let small = vec![0.1f32; 1024];
        let result = adapter.adapt(small);
        assert_eq!(result.len(), 4096);

        // Large to small (contract)
        let adapter = DimensionAdapter::new(4096, 1024);
        let large = vec![0.1f32; 4096];
        let result = adapter.adapt(large);
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn test_no_adaptation_needed() {
        let adapter = DimensionAdapter::new(4096, 4096);
        assert!(!adapter.needs_adaptation());

        let embedding = vec![0.1f32; 4096];
        let result = adapter.adapt(embedding.clone());
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_average_reduction_preserves_info() {
        let adapter = DimensionAdapter::new(4096, 2048);

        // Create embedding with distinct values
        let large: Vec<f32> = (0..4096).map(|i| i as f32 / 4096.0).collect();
        let contracted = adapter.contract(large);

        assert_eq!(contracted.len(), 2048);
        // Averaged values should be between min and max of source chunks
        // After normalization, should be unit length
        let norm: f32 = contracted.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
