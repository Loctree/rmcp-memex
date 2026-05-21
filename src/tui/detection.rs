//! Auto-detection module for embedding providers.
//!
//! Detects Ollama, MLX server, and other embedding providers automatically.

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Detected embedding provider information.
#[derive(Debug, Clone)]
pub struct DetectedProvider {
    /// Provider type
    pub kind: ProviderKind,
    /// Base URL where provider is running
    pub base_url: String,
    /// Port number
    pub port: u16,
    /// List of available models (if detected)
    pub models: Vec<String>,
    /// Suggested model to use
    pub suggested_model: Option<String>,
    /// Connection status
    pub status: ProviderStatus,
}

impl DetectedProvider {
    /// Get a human-readable description
    pub fn summary_line(&self) -> String {
        match &self.status {
            ProviderStatus::Online(model) => {
                format!("{} at {} - {}", self.kind.label(), self.base_url, model)
            }
            ProviderStatus::OnlineNoModel => {
                format!(
                    "{} at {} (no embedding model)",
                    self.kind.label(),
                    self.base_url
                )
            }
            ProviderStatus::Offline => {
                format!("{} at {} (offline)", self.kind.label(), self.base_url)
            }
        }
    }

    /// Get provider summary for UI
    pub fn summary(&self) -> String {
        self.summary_line()
    }

    /// Check if provider is usable
    pub fn is_usable(&self) -> bool {
        matches!(self.status, ProviderStatus::Online(_))
    }

    /// Get the model to use
    pub fn model(&self) -> Option<&str> {
        if let ProviderStatus::Online(ref model) = self.status {
            Some(model.as_str())
        } else {
            self.suggested_model.as_deref()
        }
    }
}

fn looks_like_embedding_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("embedding")
        || model.contains("embed")
        || model.contains("bge")
        || model.contains("nomic")
        || model.contains("mxbai")
        || model.contains("minilm")
}

fn pick_embedding_model(models: &[String]) -> Option<String> {
    models
        .iter()
        .find(|m| looks_like_embedding_model(m))
        .cloned()
}

/// Type of embedding provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// Ollama with embedding models
    Ollama,
    /// MLX embedding server
    Mlx,
    /// Generic OpenAI-compatible endpoint
    OpenAICompat,
    /// Manual configuration
    Manual,
}

impl ProviderKind {
    pub fn label(&self) -> &'static str {
        match self {
            ProviderKind::Ollama => "Ollama",
            ProviderKind::Mlx => "MLX Server",
            ProviderKind::OpenAICompat => "OpenAI-Compatible",
            ProviderKind::Manual => "Manual",
        }
    }
}

/// Provider connection status.
#[derive(Debug, Clone)]
pub enum ProviderStatus {
    /// Provider is online with a usable model
    Online(String),
    /// Provider is online but no embedding model found
    OnlineNoModel,
    /// Provider is offline
    Offline,
}

/// Response from Ollama /api/tags endpoint.
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

/// Response from OpenAI-compatible /v1/models endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Detect embedding providers on standard ports.
pub async fn detect_providers() -> Vec<DetectedProvider> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let mut providers = Vec::new();

    // Check Ollama on localhost:11434
    if let Some(provider) = detect_ollama(&client, "http://localhost", 11434).await {
        providers.push(provider);
    }

    // Check MLX on localhost:12345
    if let Some(provider) = detect_mlx(&client, "http://localhost", 12345).await {
        providers.push(provider);
    }

    // Check dragon:12345 (common remote MLX server)
    if let Some(provider) = detect_mlx(&client, "http://dragon", 12345).await {
        providers.push(provider);
    }

    providers
}

/// Check if a URL is reachable (simple health check).
/// Used for quick provider connectivity verification.
pub async fn check_health(url: &str) -> bool {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    client.get(url).send().await.is_ok()
}

/// Detect Ollama on a given host/port.
async fn detect_ollama(client: &Client, host: &str, port: u16) -> Option<DetectedProvider> {
    let base_url = format!("{}:{}", host, port);
    let tags_url = format!("{}/api/tags", base_url);

    // Try to get list of models
    let response = match client.get(&tags_url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => {
            return Some(DetectedProvider {
                kind: ProviderKind::Ollama,
                base_url: base_url.clone(),
                port,
                models: vec![],
                suggested_model: None,
                status: ProviderStatus::Offline,
            });
        }
    };

    let tags: OllamaTagsResponse = match response.json().await {
        Ok(t) => t,
        Err(_) => {
            return Some(DetectedProvider {
                kind: ProviderKind::Ollama,
                base_url,
                port,
                models: vec![],
                suggested_model: None,
                status: ProviderStatus::OnlineNoModel,
            });
        }
    };

    let models: Vec<String> = tags.models.iter().map(|m| m.name.clone()).collect();

    let embedding_model = pick_embedding_model(&models);

    let status = if let Some(ref model) = embedding_model {
        ProviderStatus::Online(model.clone())
    } else {
        ProviderStatus::OnlineNoModel
    };

    Some(DetectedProvider {
        kind: ProviderKind::Ollama,
        base_url,
        port,
        models,
        suggested_model: embedding_model,
        status,
    })
}

/// Detect MLX embedding server on a given host/port.
async fn detect_mlx(client: &Client, host: &str, port: u16) -> Option<DetectedProvider> {
    let base_url = format!("{}:{}", host, port);
    let models_url = format!("{}/v1/models", base_url);

    // Try OpenAI-compatible endpoint
    let response = match client.get(&models_url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => {
            // Don't report offline MLX servers unless we were explicitly looking for them
            return None;
        }
    };

    let models_resp: ModelsResponse = match response.json().await {
        Ok(m) => m,
        Err(_) => {
            return Some(DetectedProvider {
                kind: ProviderKind::Mlx,
                base_url,
                port,
                models: vec![],
                suggested_model: None,
                status: ProviderStatus::OnlineNoModel,
            });
        }
    };

    let models: Vec<String> = models_resp.data.iter().map(|m| m.id.clone()).collect();

    let embedding_model = pick_embedding_model(&models);

    let status = if let Some(ref model) = embedding_model {
        ProviderStatus::Online(model.clone())
    } else {
        ProviderStatus::OnlineNoModel
    };

    Some(DetectedProvider {
        kind: ProviderKind::Mlx,
        base_url,
        port,
        models,
        suggested_model: embedding_model,
        status,
    })
}

/// Check a custom endpoint for OpenAI-compatibility.
pub async fn check_custom_endpoint(url: &str) -> Result<DetectedProvider> {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_secs(3))
        .build()?;

    let base_url = url.trim_end_matches('/');

    // Extract port from URL
    let port = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|u| u.port())
        .unwrap_or(80);

    // Try /v1/models first
    let models_url = format!("{}/v1/models", base_url);
    if let Ok(resp) = client.get(&models_url).send().await {
        if resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            if let Ok(models_resp) = serde_json::from_str::<ModelsResponse>(&body) {
                let models: Vec<String> = models_resp.data.iter().map(|m| m.id.clone()).collect();

                let embedding_model = models
                    .iter()
                    .find(|m| m.contains("embedding") || m.contains("Embedding"))
                    .cloned();

                let status = if let Some(ref model) = embedding_model {
                    ProviderStatus::Online(model.clone())
                } else {
                    ProviderStatus::OnlineNoModel
                };

                return Ok(DetectedProvider {
                    kind: ProviderKind::OpenAICompat,
                    base_url: base_url.to_string(),
                    port,
                    models,
                    suggested_model: embedding_model,
                    status,
                });
            } else {
                tracing::debug!(
                    "Failed to parse /v1/models response: {}",
                    &body[..body.len().min(200)]
                );
            }
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::debug!(
                "OpenAI endpoint returned HTTP {}: {}",
                status,
                &body[..body.len().min(200)]
            );
        }
    }

    // Try Ollama endpoint
    let tags_url = format!("{}/api/tags", base_url);
    if let Ok(resp) = client.get(&tags_url).send().await {
        if resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            if let Ok(tags) = serde_json::from_str::<OllamaTagsResponse>(&body) {
                let models: Vec<String> = tags.models.iter().map(|m| m.name.clone()).collect();

                let embedding_model = pick_embedding_model(&models);

                let status = if let Some(ref model) = embedding_model {
                    ProviderStatus::Online(model.clone())
                } else if !models.is_empty() {
                    ProviderStatus::OnlineNoModel
                } else {
                    ProviderStatus::Offline
                };

                return Ok(DetectedProvider {
                    kind: ProviderKind::Ollama,
                    base_url: base_url.to_string(),
                    port,
                    models,
                    suggested_model: embedding_model,
                    status,
                });
            } else {
                tracing::debug!(
                    "Failed to parse /api/tags response: {}",
                    &body[..body.len().min(200)]
                );
            }
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::debug!(
                "Ollama endpoint returned HTTP {}: {}",
                status,
                &body[..body.len().min(200)]
            );
        }
    }

    Ok(DetectedProvider {
        kind: ProviderKind::OpenAICompat,
        base_url: base_url.to_string(),
        port,
        models: vec![],
        suggested_model: None,
        status: ProviderStatus::Offline,
    })
}

/// Get dimension explanation for UI.
/// Reports the verified dimension without guessing model variants.
pub fn dimension_explanation(dim: usize) -> String {
    format!("{dim} dims — ensure all providers match this dimension")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_kind_display() {
        assert_eq!(ProviderKind::Ollama.label(), "Ollama");
        assert_eq!(ProviderKind::Mlx.label(), "MLX Server");
    }

    #[test]
    fn pick_embedding_model_finds_embedding_keyword() {
        let models = vec!["llama3:8b".to_string(), "qwen3-embedding:8b".to_string()];
        assert_eq!(
            pick_embedding_model(&models).as_deref(),
            Some("qwen3-embedding:8b")
        );
    }

    #[test]
    fn dimension_explanation_is_dynamic() {
        let explanation = dimension_explanation(1536);
        assert!(explanation.contains("1536"));
    }
}
