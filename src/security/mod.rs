//! Security module for namespace access control.
//!
//! This module provides token-based access control for namespaces.
//! Each namespace can have an associated access token that must be provided
//! when reading or writing data to that namespace.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Token prefix for namespace access tokens
const TOKEN_PREFIX: &str = "ns_";

/// Configuration for namespace security
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceSecurityConfig {
    /// Whether token-based access control is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Path to the token store file
    #[serde(default)]
    pub token_store_path: Option<String>,
}

/// Stored token information for a namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceToken {
    /// The namespace this token grants access to
    pub namespace: String,
    /// The actual token value
    pub token: String,
    /// When the token was created (Unix timestamp)
    pub created_at: u64,
    /// Optional description/label for the token
    pub description: Option<String>,
}

/// Token store for managing namespace access tokens
#[derive(Debug)]
pub struct TokenStore {
    /// Map of namespace -> token
    tokens: Arc<RwLock<HashMap<String, NamespaceToken>>>,
    /// Path to persist tokens (if any)
    store_path: Option<String>,
}

impl TokenStore {
    /// Create a new token store
    pub fn new(store_path: Option<String>) -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            store_path,
        }
    }

    /// Load tokens from persistent storage
    pub async fn load(&self) -> Result<()> {
        if let Some(path) = &self.store_path {
            let expanded = shellexpand::tilde(path).to_string();
            let path = Path::new(&expanded);

            if path.exists() {
                let contents = tokio::fs::read_to_string(path).await?;
                let loaded: HashMap<String, NamespaceToken> = serde_json::from_str(&contents)?;
                let mut tokens = self.tokens.write().await;
                *tokens = loaded;
                info!("Loaded {} namespace tokens from {}", tokens.len(), expanded);
            }
        }
        Ok(())
    }

    /// Save tokens to persistent storage
    pub async fn save(&self) -> Result<()> {
        if let Some(path) = &self.store_path {
            let expanded = shellexpand::tilde(path).to_string();
            let path = Path::new(&expanded);

            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            let tokens = self.tokens.read().await;
            let contents = serde_json::to_string_pretty(&*tokens)?;
            tokio::fs::write(path, contents).await?;
            debug!("Saved {} namespace tokens to {}", tokens.len(), expanded);
        }
        Ok(())
    }

    /// Generate a new token for a namespace
    pub fn generate_token() -> String {
        format!(
            "{}{}",
            TOKEN_PREFIX,
            Uuid::new_v4().to_string().replace("-", "")
        )
    }

    /// Create or update a token for a namespace
    pub async fn create_token(
        &self,
        namespace: &str,
        description: Option<String>,
    ) -> Result<String> {
        let token = Self::generate_token();
        let namespace_token = NamespaceToken {
            namespace: namespace.to_string(),
            token: token.clone(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            description,
        };

        {
            let mut tokens = self.tokens.write().await;
            tokens.insert(namespace.to_string(), namespace_token);
        }

        self.save().await?;
        info!("Created token for namespace '{}'", namespace);
        Ok(token)
    }

    /// Verify a token for a namespace
    pub async fn verify_token(&self, namespace: &str, token: &str) -> bool {
        let tokens = self.tokens.read().await;
        if let Some(stored) = tokens.get(namespace) {
            stored.token == token
        } else {
            // If no token is set for this namespace, access is allowed
            // (backward compatibility - namespaces without tokens are open)
            true
        }
    }

    /// Check if a namespace has a token set
    pub async fn has_token(&self, namespace: &str) -> bool {
        let tokens = self.tokens.read().await;
        tokens.contains_key(namespace)
    }

    /// Get token info for a namespace (without revealing the actual token)
    pub async fn get_token_info(&self, namespace: &str) -> Option<(u64, Option<String>)> {
        let tokens = self.tokens.read().await;
        tokens
            .get(namespace)
            .map(|t| (t.created_at, t.description.clone()))
    }

    /// Revoke (delete) a token for a namespace
    pub async fn revoke_token(&self, namespace: &str) -> Result<bool> {
        let removed = {
            let mut tokens = self.tokens.write().await;
            tokens.remove(namespace).is_some()
        };

        if removed {
            self.save().await?;
            info!("Revoked token for namespace '{}'", namespace);
        }

        Ok(removed)
    }

    /// List all namespaces that have tokens (without revealing tokens)
    pub async fn list_protected_namespaces(&self) -> Vec<(String, u64, Option<String>)> {
        let tokens = self.tokens.read().await;
        tokens
            .values()
            .map(|t| (t.namespace.clone(), t.created_at, t.description.clone()))
            .collect()
    }
}

/// Namespace access manager that combines token verification with access control.
///
/// Deprecated: Use `crate::auth::AuthManager` instead, which provides per-token
/// scopes, namespace ACL, argon2id hashing, and token rotation.
#[deprecated(
    note = "Use crate::auth::AuthManager for multi-token auth with scopes and namespace ACL"
)]
#[derive(Debug)]
pub struct NamespaceAccessManager {
    /// Token store for managing tokens
    token_store: TokenStore,
    /// Whether token-based access control is enabled
    enabled: bool,
}

#[allow(deprecated)]
impl NamespaceAccessManager {
    /// Create a new namespace access manager
    pub fn new(config: NamespaceSecurityConfig) -> Self {
        let store_path = config.token_store_path.or_else(|| {
            if config.enabled {
                Some("~/.rmcp-servers/rmcp-memex/tokens.json".to_string())
            } else {
                None
            }
        });

        Self {
            token_store: TokenStore::new(store_path),
            enabled: config.enabled,
        }
    }

    /// Initialize the access manager (load tokens from storage)
    pub async fn init(&self) -> Result<()> {
        if self.enabled {
            self.token_store.load().await?;
        }
        Ok(())
    }

    /// Check if access control is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Verify access to a namespace
    /// Returns Ok(()) if access is granted, Err if denied
    pub async fn verify_access(&self, namespace: &str, token: Option<&str>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Check if namespace has a token
        if !self.token_store.has_token(namespace).await {
            // No token set for this namespace - allow access
            return Ok(());
        }

        // Namespace has a token - verify it
        match token {
            Some(t) => {
                if self.token_store.verify_token(namespace, t).await {
                    Ok(())
                } else {
                    warn!("Invalid token provided for namespace '{}'", namespace);
                    Err(anyhow!(
                        "Access denied: invalid token for namespace '{}'",
                        namespace
                    ))
                }
            }
            None => {
                warn!("No token provided for protected namespace '{}'", namespace);
                Err(anyhow!(
                    "Access denied: namespace '{}' requires a token. Use namespace_create_token to generate one.",
                    namespace
                ))
            }
        }
    }

    /// Create a token for a namespace
    pub async fn create_token(
        &self,
        namespace: &str,
        description: Option<String>,
    ) -> Result<String> {
        self.token_store.create_token(namespace, description).await
    }

    /// Revoke a token for a namespace
    pub async fn revoke_token(&self, namespace: &str) -> Result<bool> {
        self.token_store.revoke_token(namespace).await
    }

    /// List protected namespaces
    pub async fn list_protected_namespaces(&self) -> Vec<(String, u64, Option<String>)> {
        self.token_store.list_protected_namespaces().await
    }

    /// Get token info for a namespace
    pub async fn get_token_info(&self, namespace: &str) -> Option<(u64, Option<String>)> {
        self.token_store.get_token_info(namespace).await
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_generation() {
        let token = TokenStore::generate_token();
        assert!(token.starts_with(TOKEN_PREFIX));
        assert!(token.len() > TOKEN_PREFIX.len());
    }

    #[tokio::test]
    async fn test_token_store_create_and_verify() {
        let store = TokenStore::new(None);

        let token = store
            .create_token("test_namespace", Some("Test token".to_string()))
            .await
            .unwrap();

        assert!(store.verify_token("test_namespace", &token).await);
        assert!(!store.verify_token("test_namespace", "wrong_token").await);
        assert!(store.verify_token("other_namespace", "any_token").await); // No token set
    }

    #[tokio::test]
    async fn test_access_manager_disabled() {
        let config = NamespaceSecurityConfig::default();
        let manager = NamespaceAccessManager::new(config);

        // When disabled, all access should be allowed
        assert!(manager.verify_access("any_namespace", None).await.is_ok());
    }

    #[tokio::test]
    async fn test_access_manager_enabled() {
        let config = NamespaceSecurityConfig {
            enabled: true,
            token_store_path: None,
        };
        let manager = NamespaceAccessManager::new(config);

        // Create a token for a namespace
        let token = manager
            .create_token("protected", Some("Test".to_string()))
            .await
            .unwrap();

        // Access without token should fail
        assert!(manager.verify_access("protected", None).await.is_err());

        // Access with wrong token should fail
        assert!(
            manager
                .verify_access("protected", Some("wrong"))
                .await
                .is_err()
        );

        // Access with correct token should succeed
        assert!(
            manager
                .verify_access("protected", Some(&token))
                .await
                .is_ok()
        );

        // Unprotected namespace should allow access without token
        assert!(manager.verify_access("unprotected", None).await.is_ok());
    }
}
