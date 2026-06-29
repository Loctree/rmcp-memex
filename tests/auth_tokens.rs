//! Integration tests for Track C: Multi-Token + Namespace ACL auth system.
//!
//! Tests cover:
//! - Token creation + argon2id hash round-trip
//! - Scope enforcement (read-only rejected on write)
//! - Namespace ACL (token for kb:claude rejected on kb:reports)
//! - Token expiry -> 401
//! - Token revocation -> 401
//! - Token rotation (old -> rejected, new -> accepted)
//! - v1 -> v2 migration preserves access
//! - Persistence across load/save cycles

use rmcp_memex::auth::{AuthDenial, AuthManager, Scope, TokenStoreFile, TokenStoreV2};
use std::path::Path;

/// Helper: create a temp store path
fn temp_store() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens.json").to_str().unwrap().to_string();
    (dir, path)
}

#[tokio::test]
async fn token_create_hash_roundtrip() {
    let (_dir, path) = temp_store();
    let store = TokenStoreFile::new(path);

    let plaintext = store
        .create_token(
            "roundtrip-test".to_string(),
            vec![Scope::Read, Scope::Write],
            vec!["kb:claude".to_string(), "kb:notes".to_string()],
            None,
            "Round-trip test".to_string(),
        )
        .await
        .unwrap();

    // Token starts with expected prefix
    assert!(plaintext.starts_with("memex_"));

    // Lookup by plaintext succeeds
    let entry = store.lookup_by_plaintext(&plaintext).await.unwrap();
    assert_eq!(entry.id, "roundtrip-test");
    assert_eq!(entry.scopes, vec![Scope::Read, Scope::Write]);
    assert_eq!(
        entry.namespaces,
        vec!["kb:claude".to_string(), "kb:notes".to_string()]
    );

    // Hash is an argon2id string (not plaintext)
    assert!(entry.token_hash.contains("$argon2id$"));
    assert_ne!(entry.token_hash, plaintext);
}

#[tokio::test]
async fn scope_enforcement_read_only_rejected_on_write() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    let token = manager
        .create_token(
            "read-only".to_string(),
            vec![Scope::Read],
            vec!["*".to_string()],
            None,
            "Read-only test".to_string(),
        )
        .await
        .unwrap();

    // Read -> accepted
    assert!(manager.authorize(&token, &Scope::Read, None).await.is_ok());

    // Write -> rejected with InsufficientScope
    let err = manager
        .authorize(&token, &Scope::Write, None)
        .await
        .unwrap_err();
    match err {
        AuthDenial::InsufficientScope {
            id,
            required,
            granted,
        } => {
            assert_eq!(id, "read-only");
            assert_eq!(required, Scope::Write);
            assert_eq!(granted, vec![Scope::Read]);
        }
        other => panic!("Expected InsufficientScope, got: {:?}", other),
    }
}

#[tokio::test]
async fn namespace_acl_claude_rejected_on_reports() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    let token = manager
        .create_token(
            "claude-only".to_string(),
            vec![Scope::Read, Scope::Write],
            vec!["kb:claude".to_string()],
            None,
            "Claude namespace only".to_string(),
        )
        .await
        .unwrap();

    // kb:claude -> accepted
    assert!(
        manager
            .authorize(&token, &Scope::Read, Some("kb:claude"))
            .await
            .is_ok()
    );

    // kb:reports -> rejected with NamespaceDenied
    let err = manager
        .authorize(&token, &Scope::Read, Some("kb:reports"))
        .await
        .unwrap_err();
    match err {
        AuthDenial::NamespaceDenied { id, requested, .. } => {
            assert_eq!(id, "claude-only");
            assert_eq!(requested, "kb:reports");
        }
        other => panic!("Expected NamespaceDenied, got: {:?}", other),
    }
}

#[tokio::test]
async fn expired_token_returns_401() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    use chrono::{DateTime, Utc};

    // Create a token with an already-expired expiry timestamp
    let real_token = manager
        .create_token(
            "will-expire".to_string(),
            vec![Scope::Read],
            vec!["*".to_string()],
            Some(
                DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            "Already expired".to_string(),
        )
        .await
        .unwrap();

    // Should fail with Expired
    let err = manager.authenticate(&real_token).await.unwrap_err();
    match err {
        AuthDenial::Expired { id } => assert_eq!(id, "will-expire"),
        other => panic!("Expected Expired, got: {:?}", other),
    }
}

#[tokio::test]
async fn revoked_token_returns_401() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    let token = manager
        .create_token(
            "revoke-me".to_string(),
            vec![Scope::Read],
            vec!["*".to_string()],
            None,
            "Will be revoked".to_string(),
        )
        .await
        .unwrap();

    // Works before revocation
    assert!(manager.authenticate(&token).await.is_ok());

    // Revoke
    assert!(manager.revoke_token("revoke-me").await.unwrap());

    // Now fails
    let err = manager.authenticate(&token).await.unwrap_err();
    assert!(matches!(err, AuthDenial::InvalidToken));
}

#[tokio::test]
async fn token_rotation_old_rejected_new_accepted() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    let old_token = manager
        .create_token(
            "rotate-test".to_string(),
            vec![Scope::Read, Scope::Write],
            vec!["kb:claude".to_string()],
            None,
            "Rotation target".to_string(),
        )
        .await
        .unwrap();

    // Old works
    assert!(manager.authenticate(&old_token).await.is_ok());

    // Rotate
    let new_token = manager.rotate_token("rotate-test").await.unwrap();
    assert_ne!(old_token, new_token);

    // Old no longer works
    assert!(matches!(
        manager.authenticate(&old_token).await.unwrap_err(),
        AuthDenial::InvalidToken
    ));

    // New works
    let result = manager.authenticate(&new_token).await.unwrap();
    assert_eq!(result.token.id, "rotate-test");
}

#[tokio::test]
async fn v1_migration_preserves_access() {
    let dir = tempfile::tempdir().unwrap();
    let store_path = dir.path().join("tokens.json");

    // Write a v1 store (HashMap<String, TokenEntryV1>)
    let v1_data: std::collections::HashMap<String, serde_json::Value> = [
        (
            "kb:claude".to_string(),
            serde_json::json!({
                "namespace": "kb:claude",
                "token": "ns_migration_test_token",
                "created_at": 1700000000_u64,
                "description": "Claude namespace token"
            }),
        ),
        (
            "kb:notes".to_string(),
            serde_json::json!({
                "namespace": "kb:notes",
                "token": "ns_notes_token",
                "created_at": 1700000001_u64,
                "description": null
            }),
        ),
    ]
    .into_iter()
    .collect();

    tokio::fs::write(&store_path, serde_json::to_string_pretty(&v1_data).unwrap())
        .await
        .unwrap();

    // Load -> triggers migration
    let store = TokenStoreFile::new(store_path.to_str().unwrap().to_string());
    store.load().await.unwrap();

    // Verify v2 tokens were created
    let tokens = store.list_tokens().await;
    assert_eq!(tokens.len(), 2);

    // Old plaintext tokens should still verify via argon2
    let claude_entry = store
        .lookup_by_plaintext("ns_migration_test_token")
        .await
        .unwrap();
    assert_eq!(claude_entry.id, "migrated-kb:claude");
    assert_eq!(claude_entry.namespaces, vec!["kb:claude".to_string()]);
    // Migrated tokens get wildcard scopes
    assert!(claude_entry.scopes.contains(&Scope::Admin));

    let notes_entry = store.lookup_by_plaintext("ns_notes_token").await.unwrap();
    assert_eq!(notes_entry.id, "migrated-kb:notes");

    // v1 backup should exist
    let backup_path = format!("{}.v1.bak", store_path.to_str().unwrap());
    assert!(Path::new(&backup_path).exists());

    // Verify the persisted file is v2
    let contents = tokio::fs::read_to_string(&store_path).await.unwrap();
    let parsed: TokenStoreV2 = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed.version, 2);
}

#[tokio::test]
async fn persistence_across_load_save() {
    let (_dir, path) = temp_store();

    // Create tokens with first store instance
    let store1 = TokenStoreFile::new(path.clone());
    let token1 = store1
        .create_token(
            "persist-1".to_string(),
            vec![Scope::Read],
            vec!["ns1".to_string()],
            None,
            "First token".to_string(),
        )
        .await
        .unwrap();
    let token2 = store1
        .create_token(
            "persist-2".to_string(),
            vec![Scope::Write],
            vec!["ns2".to_string()],
            None,
            "Second token".to_string(),
        )
        .await
        .unwrap();

    // Load from fresh instance
    let store2 = TokenStoreFile::new(path);
    store2.load().await.unwrap();

    // Both tokens should be found
    assert!(store2.lookup_by_plaintext(&token1).await.is_some());
    assert!(store2.lookup_by_plaintext(&token2).await.is_some());

    // List should show both
    let tokens = store2.list_tokens().await;
    assert_eq!(tokens.len(), 2);
}

#[tokio::test]
async fn legacy_auth_token_wildcard_access() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, Some("my-legacy-secret".to_string()));
    manager.init().await.unwrap();

    // Legacy token has full wildcard access
    let result = manager
        .authorize("my-legacy-secret", &Scope::Admin, Some("any-namespace"))
        .await
        .unwrap();
    assert_eq!(result.token.id, "__legacy__");

    // Wrong token still fails
    assert!(matches!(
        manager.authenticate("wrong").await.unwrap_err(),
        AuthDenial::InvalidToken
    ));
}

#[tokio::test]
async fn admin_scope_implies_read_write() {
    let (_dir, path) = temp_store();
    let manager = AuthManager::new(path, None);
    manager.init().await.unwrap();

    let token = manager
        .create_token(
            "admin-test".to_string(),
            vec![Scope::Admin],
            vec!["*".to_string()],
            None,
            "Admin token".to_string(),
        )
        .await
        .unwrap();

    // Admin can do everything
    assert!(manager.authorize(&token, &Scope::Read, None).await.is_ok());
    assert!(manager.authorize(&token, &Scope::Write, None).await.is_ok());
    assert!(manager.authorize(&token, &Scope::Admin, None).await.is_ok());
}

#[tokio::test]
async fn duplicate_id_rejected() {
    let (_dir, path) = temp_store();
    let store = TokenStoreFile::new(path);

    store
        .create_token(
            "unique-id".to_string(),
            vec![Scope::Read],
            vec!["*".to_string()],
            None,
            "First".to_string(),
        )
        .await
        .unwrap();

    // Duplicate should fail
    let result = store
        .create_token(
            "unique-id".to_string(),
            vec![Scope::Read],
            vec!["*".to_string()],
            None,
            "Second".to_string(),
        )
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}
