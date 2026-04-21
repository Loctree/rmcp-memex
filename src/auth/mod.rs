//! Multi-token auth with per-token scopes and namespace ACL.
//!
//! Replaces the single global bearer token with a flexible token store.
//! Each token is hashed with argon2id at rest. Plaintext is shown ONCE
//! on creation and never stored.
//!
//! Vibecrafted with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use argon2::Argon2;
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ============================================================================
// Scope
// ============================================================================

/// Permission scope for a token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Read,
    Write,
    Admin,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scope::Read => write!(f, "read"),
            Scope::Write => write!(f, "write"),
            Scope::Admin => write!(f, "admin"),
        }
    }
}

impl FromStr for Scope {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "read" => Ok(Scope::Read),
            "write" => Ok(Scope::Write),
            "admin" => Ok(Scope::Admin),
            other => Err(anyhow!(
                "Unknown scope '{}'. Use: read, write, admin",
                other
            )),
        }
    }
}

// ============================================================================
// TokenEntry
// ============================================================================

/// A single token entry persisted in tokens.json (v2 schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    /// Human-readable identifier (e.g., "monika-iphone")
    pub id: String,
    /// Argon2id hash of the token. Plaintext never stored.
    pub token_hash: String,
    /// Permission scopes granted to this token.
    pub scopes: Vec<Scope>,
    /// Namespace ACL. `["*"]` means all namespaces.
    pub namespaces: Vec<String>,
    /// Optional expiry timestamp. `None` = never expires.
    pub expires_at: Option<DateTime<Utc>>,
    /// Human-readable description.
    pub description: String,
    /// When this token was created.
    pub created_at: DateTime<Utc>,
}

impl TokenEntry {
    /// Check if the token has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            Utc::now() > exp
        } else {
            false
        }
    }

    /// Check if the token grants access to a given namespace.
    pub fn has_namespace_access(&self, namespace: &str) -> bool {
        self.namespaces
            .iter()
            .any(|ns| ns == "*" || ns == namespace)
    }

    /// Check if the token has a given scope.
    pub fn has_scope(&self, scope: &Scope) -> bool {
        // Admin implies all scopes
        self.scopes.contains(&Scope::Admin) || self.scopes.contains(scope)
    }
}

// ============================================================================
// TokenStoreV2
// ============================================================================

/// Version 2 token store schema, persisted as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStoreV2 {
    pub version: u32,
    pub tokens: Vec<TokenEntry>,
}

impl Default for TokenStoreV2 {
    fn default() -> Self {
        Self {
            version: 2,
            tokens: Vec::new(),
        }
    }
}

/// Version 1 schema (legacy) for migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenEntryV1 {
    namespace: String,
    token: String,
    created_at: u64,
    description: Option<String>,
}

/// Persistent token store backed by `tokens.json`.
#[derive(Debug)]
pub struct TokenStoreFile {
    store: Arc<RwLock<TokenStoreV2>>,
    store_path: String,
}

impl TokenStoreFile {
    /// Create a new token store at the given path.
    pub fn new(store_path: String) -> Self {
        Self {
            store: Arc::new(RwLock::new(TokenStoreV2::default())),
            store_path,
        }
    }

    /// Expand and return the canonical file path.
    fn expanded_path(&self) -> String {
        shellexpand::tilde(&self.store_path).to_string()
    }

    /// Load tokens from disk. Handles v1 -> v2 migration.
    pub async fn load(&self) -> Result<()> {
        let expanded = self.expanded_path();
        let path = Path::new(&expanded);

        if !path.exists() {
            debug!("No token store at {}, starting fresh", expanded);
            return Ok(());
        }

        let contents = tokio::fs::read_to_string(path).await?;

        // Try v2 first
        if let Ok(v2) = serde_json::from_str::<TokenStoreV2>(&contents)
            && v2.version == 2
        {
            let count = v2.tokens.len();
            let mut store = self.store.write().await;
            *store = v2;
            info!("Loaded {} tokens from v2 store at {}", count, expanded);
            return Ok(());
        }

        // Try v1 (legacy: HashMap<String, TokenEntryV1>)
        if let Ok(v1_map) =
            serde_json::from_str::<std::collections::HashMap<String, TokenEntryV1>>(&contents)
        {
            info!(
                "Detected v1 token store with {} entries, migrating to v2",
                v1_map.len()
            );

            // Back up v1
            let backup_path = format!("{}.v1.bak", expanded);
            tokio::fs::copy(&expanded, &backup_path).await?;
            info!("Backed up v1 store to {}", backup_path);

            // Migrate each v1 token
            let argon2 = Argon2::default();
            let mut migrated = Vec::new();
            for (ns, entry) in &v1_map {
                let salt = SaltString::generate(&mut OsRng);
                let hash = argon2
                    .hash_password(entry.token.as_bytes(), &salt)
                    .map_err(|e| anyhow!("Failed to hash v1 token for '{}': {}", ns, e))?
                    .to_string();

                migrated.push(TokenEntry {
                    id: format!("migrated-{}", ns),
                    token_hash: hash,
                    scopes: vec![Scope::Read, Scope::Write, Scope::Admin],
                    namespaces: vec![ns.clone()],
                    expires_at: None,
                    description: entry
                        .description
                        .clone()
                        .unwrap_or_else(|| format!("Migrated from v1 for namespace '{}'", ns)),
                    created_at: DateTime::from_timestamp(entry.created_at as i64, 0)
                        .unwrap_or_else(Utc::now),
                });
            }

            let v2 = TokenStoreV2 {
                version: 2,
                tokens: migrated,
            };
            let mut store = self.store.write().await;
            *store = v2;
            drop(store);

            self.save().await?;
            warn!(
                "Migrated v1 token store to v2. Old store backed up to {}",
                backup_path
            );
            return Ok(());
        }

        Err(anyhow!(
            "Cannot parse token store at {}. Expected v2 or v1 format.",
            expanded
        ))
    }

    /// Save current store to disk.
    pub async fn save(&self) -> Result<()> {
        let expanded = self.expanded_path();
        let path = Path::new(&expanded);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let store = self.store.read().await;
        let contents = serde_json::to_string_pretty(&*store)?;
        tokio::fs::write(path, contents).await?;
        debug!("Saved {} tokens to {}", store.tokens.len(), expanded);
        Ok(())
    }

    /// Create a new token, hash it, store it, and return the plaintext.
    pub async fn create_token(
        &self,
        id: String,
        scopes: Vec<Scope>,
        namespaces: Vec<String>,
        expires_at: Option<DateTime<Utc>>,
        description: String,
    ) -> Result<String> {
        // Check for duplicate id
        {
            let store = self.store.read().await;
            if store.tokens.iter().any(|t| t.id == id) {
                return Err(anyhow!(
                    "Token with id '{}' already exists. Use 'auth revoke' first or pick a different id.",
                    id
                ));
            }
        }

        // Generate plaintext token
        let plaintext = format!("memex_{}", Uuid::new_v4().to_string().replace('-', ""));

        // Hash it
        let argon2 = Argon2::default();
        let salt = SaltString::generate(&mut OsRng);
        let hash = argon2
            .hash_password(plaintext.as_bytes(), &salt)
            .map_err(|e| anyhow!("Failed to hash token: {}", e))?
            .to_string();

        let entry = TokenEntry {
            id: id.clone(),
            token_hash: hash,
            scopes,
            namespaces,
            expires_at,
            description,
            created_at: Utc::now(),
        };

        {
            let mut store = self.store.write().await;
            store.tokens.push(entry);
        }

        self.save().await?;
        info!("Created token '{}'", id);
        Ok(plaintext)
    }

    /// List all token entries (no plaintext exposed).
    pub async fn list_tokens(&self) -> Vec<TokenEntry> {
        let store = self.store.read().await;
        store.tokens.clone()
    }

    /// Revoke (remove) a token by id.
    pub async fn revoke_token(&self, id: &str) -> Result<bool> {
        let removed = {
            let mut store = self.store.write().await;
            let before = store.tokens.len();
            store.tokens.retain(|t| t.id != id);
            store.tokens.len() < before
        };

        if removed {
            self.save().await?;
            info!("Revoked token '{}'", id);
        }
        Ok(removed)
    }

    /// Rotate a token: revoke old, create new with same metadata.
    pub async fn rotate_token(&self, id: &str) -> Result<String> {
        let old_entry = {
            let store = self.store.read().await;
            store
                .tokens
                .iter()
                .find(|t| t.id == id)
                .cloned()
                .ok_or_else(|| anyhow!("Token '{}' not found", id))?
        };

        // Remove old
        {
            let mut store = self.store.write().await;
            store.tokens.retain(|t| t.id != id);
        }

        // Create new with same metadata
        self.create_token(
            old_entry.id,
            old_entry.scopes,
            old_entry.namespaces,
            old_entry.expires_at,
            old_entry.description,
        )
        .await
    }

    /// Look up a token by verifying a plaintext against all stored hashes.
    /// Returns the matching entry if found and valid.
    pub async fn lookup_by_plaintext(&self, plaintext: &str) -> Option<TokenEntry> {
        let store = self.store.read().await;
        let argon2 = Argon2::default();

        for entry in &store.tokens {
            if let Ok(parsed_hash) = PasswordHash::new(&entry.token_hash)
                && argon2
                    .verify_password(plaintext.as_bytes(), &parsed_hash)
                    .is_ok()
            {
                return Some(entry.clone());
            }
        }
        None
    }
}

// ============================================================================
// AuthManager
// ============================================================================

/// Unified auth manager replacing the legacy `NamespaceAccessManager`.
///
/// Handles:
/// - Token lookup by hash (argon2id verification)
/// - Scope enforcement (read/write/admin)
/// - Namespace ACL checks
/// - Expiry checks
/// - Legacy `--auth-token` compatibility (mapped to wildcard token)
#[derive(Debug)]
pub struct AuthManager {
    token_store: TokenStoreFile,
    /// Legacy fallback: if set, a single token that grants wildcard access.
    legacy_token: Option<String>,
}

/// Result of authenticating a request.
#[derive(Debug, Clone)]
pub struct AuthResult {
    /// The token entry that authenticated the request.
    pub token: TokenEntry,
}

/// Reason an auth check was denied.
#[derive(Debug, Clone)]
pub enum AuthDenial {
    /// No bearer token provided.
    MissingToken,
    /// Token provided but not recognized.
    InvalidToken,
    /// Token is expired.
    Expired { id: String },
    /// Token lacks the required scope.
    InsufficientScope {
        id: String,
        required: Scope,
        granted: Vec<Scope>,
    },
    /// Token lacks access to the requested namespace.
    NamespaceDenied {
        id: String,
        requested: String,
        allowed: Vec<String>,
    },
}

impl fmt::Display for AuthDenial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthDenial::MissingToken => write!(f, "Authorization header missing or malformed"),
            AuthDenial::InvalidToken => write!(f, "Invalid or unrecognized token"),
            AuthDenial::Expired { id } => write!(f, "Token '{}' has expired", id),
            AuthDenial::InsufficientScope {
                id,
                required,
                granted,
            } => {
                let granted_str: Vec<String> = granted.iter().map(|s| s.to_string()).collect();
                write!(
                    f,
                    "Token '{}' lacks scope '{}' (has: [{}])",
                    id,
                    required,
                    granted_str.join(", ")
                )
            }
            AuthDenial::NamespaceDenied {
                id,
                requested,
                allowed,
            } => write!(
                f,
                "Token '{}' cannot access namespace '{}' (allowed: [{}])",
                id,
                requested,
                allowed.join(", ")
            ),
        }
    }
}

impl AuthManager {
    /// Create a new AuthManager with the given store path and optional legacy token.
    pub fn new(store_path: String, legacy_token: Option<String>) -> Self {
        Self {
            token_store: TokenStoreFile::new(store_path),
            legacy_token,
        }
    }

    /// Initialize: load tokens from disk, warn about legacy token usage.
    pub async fn init(&self) -> Result<()> {
        self.token_store.load().await?;

        if self.legacy_token.is_some() {
            warn!(
                "DEPRECATED: --auth-token flag used. This maps to a single wildcard token. \
                 Migrate to 'rmcp-memex auth create' for per-token scopes and namespace ACL."
            );
        }
        Ok(())
    }

    /// Authenticate a bearer token. Returns the matched entry or denial reason.
    pub async fn authenticate(&self, bearer_token: &str) -> Result<AuthResult, AuthDenial> {
        // Check legacy token first
        if let Some(ref legacy) = self.legacy_token
            && bearer_token == legacy
        {
            return Ok(AuthResult {
                token: TokenEntry {
                    id: "__legacy__".to_string(),
                    token_hash: String::new(),
                    scopes: vec![Scope::Read, Scope::Write, Scope::Admin],
                    namespaces: vec!["*".to_string()],
                    expires_at: None,
                    description: "Legacy --auth-token (wildcard)".to_string(),
                    created_at: Utc::now(),
                },
            });
        }

        // Look up in v2 store
        match self.token_store.lookup_by_plaintext(bearer_token).await {
            Some(entry) => {
                if entry.is_expired() {
                    return Err(AuthDenial::Expired {
                        id: entry.id.clone(),
                    });
                }
                Ok(AuthResult { token: entry })
            }
            None => Err(AuthDenial::InvalidToken),
        }
    }

    /// Full authorization check: authenticate + scope + namespace.
    pub async fn authorize(
        &self,
        bearer_token: &str,
        required_scope: &Scope,
        namespace: Option<&str>,
    ) -> Result<AuthResult, AuthDenial> {
        let result = self.authenticate(bearer_token).await?;

        // Check scope
        if !result.token.has_scope(required_scope) {
            return Err(AuthDenial::InsufficientScope {
                id: result.token.id.clone(),
                required: required_scope.clone(),
                granted: result.token.scopes.clone(),
            });
        }

        // Check namespace ACL (if a namespace is specified)
        if let Some(ns) = namespace
            && !result.token.has_namespace_access(ns)
        {
            return Err(AuthDenial::NamespaceDenied {
                id: result.token.id.clone(),
                requested: ns.to_string(),
                allowed: result.token.namespaces.clone(),
            });
        }

        Ok(result)
    }

    /// Delegate to token store: create a new token.
    pub async fn create_token(
        &self,
        id: String,
        scopes: Vec<Scope>,
        namespaces: Vec<String>,
        expires_at: Option<DateTime<Utc>>,
        description: String,
    ) -> Result<String> {
        self.token_store
            .create_token(id, scopes, namespaces, expires_at, description)
            .await
    }

    /// Delegate to token store: list all tokens.
    pub async fn list_tokens(&self) -> Vec<TokenEntry> {
        self.token_store.list_tokens().await
    }

    /// Delegate to token store: revoke a token.
    pub async fn revoke_token(&self, id: &str) -> Result<bool> {
        self.token_store.revoke_token(id).await
    }

    /// Delegate to token store: rotate a token.
    pub async fn rotate_token(&self, id: &str) -> Result<String> {
        self.token_store.rotate_token(id).await
    }

    /// Check if any tokens are configured (v2 or legacy).
    pub async fn has_any_tokens(&self) -> bool {
        self.legacy_token.is_some() || !self.token_store.list_tokens().await.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_display_and_parse() {
        assert_eq!(Scope::Read.to_string(), "read");
        assert_eq!(Scope::Write.to_string(), "write");
        assert_eq!(Scope::Admin.to_string(), "admin");

        assert_eq!(Scope::from_str("read").unwrap(), Scope::Read);
        assert_eq!(Scope::from_str("WRITE").unwrap(), Scope::Write);
        assert_eq!(Scope::from_str("Admin").unwrap(), Scope::Admin);
        assert!(Scope::from_str("invalid").is_err());
    }

    #[test]
    fn token_entry_scope_check() {
        let entry = TokenEntry {
            id: "test".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["ns1".to_string()],
            expires_at: None,
            description: "test".to_string(),
            created_at: Utc::now(),
        };

        assert!(entry.has_scope(&Scope::Read));
        assert!(!entry.has_scope(&Scope::Write));
        assert!(!entry.has_scope(&Scope::Admin));
    }

    #[test]
    fn admin_scope_implies_all() {
        let entry = TokenEntry {
            id: "admin".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Admin],
            namespaces: vec!["*".to_string()],
            expires_at: None,
            description: "admin".to_string(),
            created_at: Utc::now(),
        };

        assert!(entry.has_scope(&Scope::Read));
        assert!(entry.has_scope(&Scope::Write));
        assert!(entry.has_scope(&Scope::Admin));
    }

    #[test]
    fn namespace_wildcard_access() {
        let entry = TokenEntry {
            id: "wild".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["*".to_string()],
            expires_at: None,
            description: "wildcard".to_string(),
            created_at: Utc::now(),
        };

        assert!(entry.has_namespace_access("kb:claude"));
        assert!(entry.has_namespace_access("anything"));
    }

    #[test]
    fn namespace_acl_check() {
        let entry = TokenEntry {
            id: "limited".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["kb:claude".to_string(), "kb:mikserka".to_string()],
            expires_at: None,
            description: "limited".to_string(),
            created_at: Utc::now(),
        };

        assert!(entry.has_namespace_access("kb:claude"));
        assert!(entry.has_namespace_access("kb:mikserka"));
        assert!(!entry.has_namespace_access("kb:reports"));
    }

    #[test]
    fn token_entry_expiry() {
        let expired = TokenEntry {
            id: "expired".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["*".to_string()],
            expires_at: Some(
                DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            description: "expired".to_string(),
            created_at: Utc::now(),
        };
        assert!(expired.is_expired());

        let future = TokenEntry {
            id: "future".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["*".to_string()],
            expires_at: Some(
                DateTime::parse_from_rfc3339("2099-12-31T00:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            description: "future".to_string(),
            created_at: Utc::now(),
        };
        assert!(!future.is_expired());

        let no_expiry = TokenEntry {
            id: "noexp".to_string(),
            token_hash: String::new(),
            scopes: vec![Scope::Read],
            namespaces: vec!["*".to_string()],
            expires_at: None,
            description: "no expiry".to_string(),
            created_at: Utc::now(),
        };
        assert!(!no_expiry.is_expired());
    }

    #[tokio::test]
    async fn token_create_and_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let store = TokenStoreFile::new(store_path);

        let plaintext = store
            .create_token(
                "test-token".to_string(),
                vec![Scope::Read, Scope::Write],
                vec!["kb:claude".to_string()],
                None,
                "Test token".to_string(),
            )
            .await
            .unwrap();

        assert!(plaintext.starts_with("memex_"));

        // Lookup should succeed
        let found = store.lookup_by_plaintext(&plaintext).await;
        assert!(found.is_some());
        let entry = found.unwrap();
        assert_eq!(entry.id, "test-token");
        assert_eq!(entry.scopes, vec![Scope::Read, Scope::Write]);

        // Wrong token should fail
        let not_found = store.lookup_by_plaintext("memex_wrong").await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn token_revoke() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let store = TokenStoreFile::new(store_path);
        let plaintext = store
            .create_token(
                "revokable".to_string(),
                vec![Scope::Read],
                vec!["*".to_string()],
                None,
                "Will be revoked".to_string(),
            )
            .await
            .unwrap();

        // Verify it works
        assert!(store.lookup_by_plaintext(&plaintext).await.is_some());

        // Revoke
        assert!(store.revoke_token("revokable").await.unwrap());

        // Should no longer be found
        assert!(store.lookup_by_plaintext(&plaintext).await.is_none());
    }

    #[tokio::test]
    async fn auth_manager_scope_enforcement() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let manager = AuthManager::new(store_path, None);
        manager.init().await.unwrap();

        let plaintext = manager
            .create_token(
                "read-only".to_string(),
                vec![Scope::Read],
                vec!["*".to_string()],
                None,
                "Read-only token".to_string(),
            )
            .await
            .unwrap();

        // Read should succeed
        let result = manager.authorize(&plaintext, &Scope::Read, None).await;
        assert!(result.is_ok());

        // Write should fail with InsufficientScope
        let result = manager.authorize(&plaintext, &Scope::Write, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AuthDenial::InsufficientScope { required, .. } => {
                assert_eq!(required, Scope::Write);
            }
            other => panic!("Expected InsufficientScope, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn auth_manager_namespace_enforcement() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let manager = AuthManager::new(store_path, None);
        manager.init().await.unwrap();

        let plaintext = manager
            .create_token(
                "ns-limited".to_string(),
                vec![Scope::Read, Scope::Write],
                vec!["kb:claude".to_string()],
                None,
                "Limited to kb:claude".to_string(),
            )
            .await
            .unwrap();

        // Allowed namespace
        let result = manager
            .authorize(&plaintext, &Scope::Read, Some("kb:claude"))
            .await;
        assert!(result.is_ok());

        // Disallowed namespace
        let result = manager
            .authorize(&plaintext, &Scope::Read, Some("kb:reports"))
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AuthDenial::NamespaceDenied { requested, .. } => {
                assert_eq!(requested, "kb:reports");
            }
            other => panic!("Expected NamespaceDenied, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn auth_manager_legacy_token() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let manager = AuthManager::new(store_path, Some("my-legacy-token".to_string()));
        manager.init().await.unwrap();

        // Legacy token should have wildcard access
        let result = manager
            .authorize("my-legacy-token", &Scope::Admin, Some("any-ns"))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().token.id, "__legacy__");

        // Wrong token should fail
        let result = manager.authenticate("wrong-token").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn auth_manager_expired_token() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let manager = AuthManager::new(store_path, None);
        manager.init().await.unwrap();

        // Directly create an expired entry via the store
        {
            let store = &manager.token_store;
            let argon2 = Argon2::default();
            let salt = SaltString::generate(&mut OsRng);
            let hash = argon2
                .hash_password(b"expired_token_value", &salt)
                .unwrap()
                .to_string();

            let entry = TokenEntry {
                id: "expired-test".to_string(),
                token_hash: hash,
                scopes: vec![Scope::Read],
                namespaces: vec!["*".to_string()],
                expires_at: Some(
                    DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                        .unwrap()
                        .with_timezone(&Utc),
                ),
                description: "Expired test".to_string(),
                created_at: Utc::now(),
            };
            let mut s = store.store.write().await;
            s.tokens.push(entry);
        }

        let result = manager.authenticate("expired_token_value").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AuthDenial::Expired { id } => assert_eq!(id, "expired-test"),
            other => panic!("Expected Expired, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn token_store_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        // Create and save
        let store1 = TokenStoreFile::new(store_path.clone());
        let plaintext = store1
            .create_token(
                "persist-test".to_string(),
                vec![Scope::Read],
                vec!["*".to_string()],
                None,
                "Persistence test".to_string(),
            )
            .await
            .unwrap();

        // Load from fresh instance
        let store2 = TokenStoreFile::new(store_path);
        store2.load().await.unwrap();

        let found = store2.lookup_by_plaintext(&plaintext).await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "persist-test");
    }

    #[tokio::test]
    async fn token_rotate() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json").to_str().unwrap().to_string();

        let store = TokenStoreFile::new(store_path);
        let old_plaintext = store
            .create_token(
                "rotate-me".to_string(),
                vec![Scope::Read, Scope::Write],
                vec!["kb:claude".to_string()],
                None,
                "Will be rotated".to_string(),
            )
            .await
            .unwrap();

        // Rotate
        let new_plaintext = store.rotate_token("rotate-me").await.unwrap();
        assert_ne!(old_plaintext, new_plaintext);

        // Old should not work
        assert!(store.lookup_by_plaintext(&old_plaintext).await.is_none());

        // New should work
        let found = store.lookup_by_plaintext(&new_plaintext).await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "rotate-me");
    }

    #[tokio::test]
    async fn v1_migration() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("tokens.json");

        // Write a v1 store
        let v1_data: std::collections::HashMap<String, serde_json::Value> = [(
            "kb:claude".to_string(),
            serde_json::json!({
                "namespace": "kb:claude",
                "token": "ns_test123456",
                "created_at": 1700000000_u64,
                "description": "Original v1 token"
            }),
        )]
        .into_iter()
        .collect();

        tokio::fs::write(&store_path, serde_json::to_string_pretty(&v1_data).unwrap())
            .await
            .unwrap();

        // Load should migrate
        let store = TokenStoreFile::new(store_path.to_str().unwrap().to_string());
        store.load().await.unwrap();

        // Verify migration
        let tokens = store.list_tokens().await;
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].id, "migrated-kb:claude");
        assert_eq!(tokens[0].namespaces, vec!["kb:claude".to_string()]);
        assert_eq!(
            tokens[0].scopes,
            vec![Scope::Read, Scope::Write, Scope::Admin]
        );

        // Old plaintext should verify
        let found = store.lookup_by_plaintext("ns_test123456").await;
        assert!(found.is_some());

        // Backup file should exist
        let backup_path = format!("{}.v1.bak", store_path.to_str().unwrap());
        assert!(Path::new(&backup_path).exists());
    }
}
