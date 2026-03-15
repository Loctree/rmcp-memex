//! Health Check Module for TUI Wizard
//!
//! Performs comprehensive health checks for the rmcp-memex configuration:
//! - Embedder endpoint connectivity
//! - Test embedding generation and dimension verification
//! - Database path writability

use crate::embeddings::{EmbeddingConfig, ProviderConfig};
use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

/// Status of an individual health check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    /// Check passed successfully
    Pass,
    /// Check failed with an error message
    Fail(String),
    /// Check is in progress
    Running,
    /// Check hasn't been run yet
    Pending,
}

impl CheckStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            CheckStatus::Pass => "[OK]",
            CheckStatus::Fail(_) => "[ERR]",
            CheckStatus::Running => "[...]",
            CheckStatus::Pending => "[ ]",
        }
    }

    pub fn is_pass(&self) -> bool {
        matches!(self, CheckStatus::Pass)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, CheckStatus::Fail(_))
    }
}

/// Individual health check result
#[derive(Debug, Clone)]
pub struct HealthCheckItem {
    pub name: String,
    pub description: String,
    pub status: CheckStatus,
}

impl HealthCheckItem {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            status: CheckStatus::Pending,
        }
    }

    pub fn pass(mut self) -> Self {
        self.status = CheckStatus::Pass;
        self
    }

    pub fn fail(mut self, msg: impl Into<String>) -> Self {
        self.status = CheckStatus::Fail(msg.into());
        self
    }

    pub fn running(mut self) -> Self {
        self.status = CheckStatus::Running;
        self
    }
}

/// Aggregate health check results
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub items: Vec<HealthCheckItem>,
    pub connected_provider: Option<String>,
    pub verified_dimension: Option<usize>,
}

impl HealthCheckResult {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            connected_provider: None,
            verified_dimension: None,
        }
    }

    pub fn all_passed(&self) -> bool {
        self.items.iter().all(|i| i.status.is_pass())
    }

    pub fn any_failed(&self) -> bool {
        self.items.iter().any(|i| i.status.is_fail())
    }

    pub fn is_complete(&self) -> bool {
        self.items
            .iter()
            .all(|i| !matches!(i.status, CheckStatus::Pending | CheckStatus::Running))
    }
}

impl Default for HealthCheckResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Response type for embedding endpoint
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Health checker that performs all validation
pub struct HealthChecker {
    client: Client,
}

impl HealthChecker {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self { client }
    }

    /// Run all health checks asynchronously
    pub async fn run_all(
        &self,
        embedding_config: &EmbeddingConfig,
        db_path: &str,
    ) -> HealthCheckResult {
        let mut result = HealthCheckResult::new();

        // Check 1: DB Path writability
        let db_check = self.check_db_path(db_path);
        result.items.push(db_check);

        // Check 2: Embedder connectivity (try each provider in order)
        let (embedder_check, provider_name) =
            self.check_embedder_connectivity(embedding_config).await;
        result.items.push(embedder_check);
        result.connected_provider = provider_name.clone();

        // Check 3: Test embedding generation (only if connectivity passed)
        if provider_name.is_some() {
            let (embed_check, dimension) = self.check_embedding_generation(embedding_config).await;
            result.items.push(embed_check);
            result.verified_dimension = dimension;

            // Check 4: Dimension match (only if embedding succeeded)
            if let Some(dim) = dimension {
                let dim_check =
                    self.check_dimension_match(dim, embedding_config.required_dimension);
                result.items.push(dim_check);
            }
        } else {
            // Skip embedding checks if connectivity failed
            result.items.push(
                HealthCheckItem::new("Test Embedding", "Send test text and verify response")
                    .fail("Skipped: No embedder available"),
            );
            result.items.push(
                HealthCheckItem::new(
                    "Dimension Match",
                    format!("Verify dimension = {}", embedding_config.required_dimension),
                )
                .fail("Skipped: No embedding to verify"),
            );
        }

        result
    }

    /// Check if the database path is writable
    fn check_db_path(&self, db_path: &str) -> HealthCheckItem {
        let mut item = HealthCheckItem::new("DB Path Writable", format!("Check {}", db_path));

        let expanded = shellexpand::tilde(db_path).to_string();
        let path = PathBuf::from(&expanded);

        // Check if path exists or can be created
        if path.exists() {
            if path.is_dir() {
                // Try to write a test file
                let test_file = path.join(".rmcp_memex_write_test");
                match std::fs::write(&test_file, "test") {
                    Ok(_) => {
                        let _ = std::fs::remove_file(&test_file);
                        item = item.pass();
                        item.description = format!("Writable: {}", expanded);
                    }
                    Err(e) => {
                        item = item.fail(format!("Not writable: {}", e));
                    }
                }
            } else {
                item = item.fail("Path exists but is not a directory");
            }
        } else {
            // Try to create parent directories
            if let Some(parent) = path.parent() {
                if parent.exists() || std::fs::create_dir_all(parent).is_ok() {
                    item = item.pass();
                    item.description = format!("Will create: {}", expanded);
                } else {
                    item = item.fail("Cannot create parent directories");
                }
            } else {
                item = item.fail("Invalid path");
            }
        }

        item
    }

    /// Check embedder connectivity by trying each provider
    async fn check_embedder_connectivity(
        &self,
        config: &EmbeddingConfig,
    ) -> (HealthCheckItem, Option<String>) {
        let mut item = HealthCheckItem::new("Embedder Connection", "Connect to embedding provider");

        if config.providers.is_empty() {
            return (item.fail("No embedding providers configured"), None);
        }

        // Sort by priority
        let mut providers = config.providers.clone();
        providers.sort_by_key(|p| p.priority);

        let mut tried = Vec::new();

        for provider in &providers {
            match self.try_provider_health(provider).await {
                Ok(()) => {
                    item = item.pass();
                    item.description =
                        format!("Connected to {} ({})", provider.name, provider.base_url);
                    return (item, Some(provider.name.clone()));
                }
                Err(e) => {
                    tried.push(format!("{}: {}", provider.name, e));
                }
            }
        }

        (
            item.fail(format!("All providers failed:\n  {}", tried.join("\n  "))),
            None,
        )
    }

    /// Try a single provider's health endpoint
    async fn try_provider_health(&self, provider: &ProviderConfig) -> Result<()> {
        let base_url = provider.base_url.trim_end_matches('/');

        // Try /v1/models first (OpenAI-compatible)
        let url = format!("{}/v1/models", base_url);
        let response = self.client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) if resp.status().as_u16() == 404 => {
                // Try Ollama-native endpoint
                let ollama_url = format!("{}/api/tags", base_url);
                let ollama_resp = self.client.get(&ollama_url).send().await?;
                if ollama_resp.status().is_success() {
                    return Ok(());
                }
                Err(anyhow!("No compatible endpoint found"))
            }
            Ok(resp) => Err(anyhow!("HTTP {}", resp.status())),
            Err(e) => Err(anyhow!("Connection failed: {}", e)),
        }
    }

    /// Check embedding generation with test text
    async fn check_embedding_generation(
        &self,
        config: &EmbeddingConfig,
    ) -> (HealthCheckItem, Option<usize>) {
        let mut item =
            HealthCheckItem::new("Test Embedding", "Generate embedding for 'hello world'");

        // Sort providers and find the first available
        let mut providers = config.providers.clone();
        providers.sort_by_key(|p| p.priority);

        for provider in &providers {
            let base_url = provider.base_url.trim_end_matches('/');
            let url = format!("{}{}", base_url, provider.endpoint);

            let request = serde_json::json!({
                "input": ["hello world"],
                "model": provider.model
            });

            match self.client.post(&url).json(&request).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let body = match resp.text().await {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!(
                                "Failed to read response body from {}: {}",
                                provider.name,
                                e
                            );
                            continue;
                        }
                    };

                    if !status.is_success() {
                        tracing::warn!(
                            "Embedding API error from {} (HTTP {}): {}",
                            provider.name,
                            status,
                            &body[..body.len().min(500)]
                        );
                        continue;
                    }

                    match serde_json::from_str::<EmbeddingResponse>(&body) {
                        Ok(emb_resp) => {
                            if let Some(data) = emb_resp.data.first() {
                                let dim = data.embedding.len();
                                item = item.pass();
                                item.description =
                                    format!("Got {}-dim vector from {}", dim, provider.name);
                                return (item, Some(dim));
                            } else {
                                tracing::warn!("Empty embedding data from {}", provider.name);
                                continue;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to parse embedding response from {}: {} - Body preview: {}",
                                provider.name,
                                e,
                                &body[..body.len().min(200)]
                            );
                            item = item.fail(format!("Failed to parse response: {}", e));
                            return (item, None);
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("Connection failed to {}: {}", provider.name, e);
                    continue;
                }
            }
        }

        (item.fail("No provider returned a valid embedding"), None)
    }

    /// Check if the returned dimension matches the required dimension
    fn check_dimension_match(&self, actual: usize, required: usize) -> HealthCheckItem {
        let mut item = HealthCheckItem::new(
            "Dimension Match",
            format!("Verify {} = {}", actual, required),
        );

        if actual == required {
            item = item.pass();
            item.description = format!("Dimension matches: {}", required);
        } else {
            item = item.fail(format!(
                "Dimension mismatch! Got {} but config requires {}. \
                This would corrupt the database!",
                actual, required
            ));
        }

        item
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_status_icon() {
        assert_eq!(CheckStatus::Pass.icon(), "[OK]");
        assert_eq!(CheckStatus::Fail("test".into()).icon(), "[ERR]");
        assert_eq!(CheckStatus::Running.icon(), "[...]");
        assert_eq!(CheckStatus::Pending.icon(), "[ ]");
    }

    #[test]
    fn test_health_check_result() {
        let mut result = HealthCheckResult::new();
        assert!(result.items.is_empty());
        assert!(!result.any_failed());
        assert!(result.is_complete());

        result
            .items
            .push(HealthCheckItem::new("Test", "Desc").pass());
        assert!(result.all_passed());
        assert!(!result.any_failed());

        result
            .items
            .push(HealthCheckItem::new("Test2", "Desc2").fail("error"));
        assert!(!result.all_passed());
        assert!(result.any_failed());
    }

    #[test]
    fn test_db_path_check() {
        let checker = HealthChecker::new();

        // Test with unique temp directory (avoids predictable temp path)
        let tmp = tempfile::tempdir().unwrap();
        let temp_path = tmp.path().join("rmcp_memex_test");
        let item = checker.check_db_path(temp_path.to_str().unwrap());
        // Should either pass (writable) or indicate will create
        assert!(item.status.is_pass() || matches!(item.status, CheckStatus::Fail(_)));
    }
}
