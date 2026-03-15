use crate::{
    EmbeddingConfig, MLXBridge, ProviderConfig, SliceMode, compute_content_hash, rag::RAGPipeline,
    storage::StorageManager,
};
use anyhow::{Result, anyhow};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Try to connect to MLX server for tests. Returns None if unavailable.
/// Tests that need embeddings should skip if this returns None.
async fn try_mlx_bridge() -> Option<Arc<Mutex<MLXBridge>>> {
    // Use test config - try localhost only, fail fast
    let config = EmbeddingConfig {
        required_dimension: 4096,
        max_batch_chars: 32000,
        max_batch_items: 16,
        providers: vec![ProviderConfig {
            name: "test-local".to_string(),
            base_url: "http://localhost:12345".to_string(),
            model: "test-model".to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        ..Default::default()
    };

    match MLXBridge::new(&config).await {
        Ok(bridge) => Some(Arc::new(Mutex::new(bridge))),
        Err(_) => None,
    }
}

/// Macro to skip test if MLX is unavailable
macro_rules! require_mlx {
    ($mlx:expr) => {
        match $mlx {
            Some(bridge) => bridge,
            None => {
                eprintln!("⚠ Skipping test: MLX server unavailable at localhost:12345");
                return Ok(());
            }
        }
    };
}

#[tokio::test]
async fn memory_roundtrip_and_search() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    let storage = Arc::new(StorageManager::new(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;

    // Index a memory chunk with explicit Flat mode so ID is preserved
    let returned_id = rag
        .index_text_with_mode(
            Some("testns"),
            "doc1".to_string(),
            "Ala ma kota".to_string(),
            json!({"lang": "pl"}),
            SliceMode::Flat,
        )
        .await?;

    // ID should be unchanged with Flat mode
    assert_eq!(returned_id, "doc1");

    // Read it back
    let fetched = rag
        .memory_get("testns", "doc1")
        .await?
        .ok_or_else(|| anyhow!("doc missing"))?;
    assert_eq!(fetched.text, "Ala ma kota");
    assert_eq!(fetched.namespace, "testns");

    // Semantic search within namespace
    let results = rag.memory_search("testns", "kota", 1).await?;
    assert!(!results.is_empty(), "expected at least one search result");
    assert_eq!(results[0].namespace, "testns");

    Ok(())
}

#[test]
fn test_content_hash_deterministic() {
    let content = "Test content for hashing";
    let hash1 = compute_content_hash(content);
    let hash2 = compute_content_hash(content);

    // Same content should produce identical hash
    assert_eq!(hash1, hash2);

    // Hash should be 64 characters (SHA256 hex)
    assert_eq!(hash1.len(), 64);

    // Different content should produce different hash
    let hash3 = compute_content_hash("Different content");
    assert_ne!(hash1, hash3);
}

#[test]
fn test_content_hash_slight_difference() {
    // Even a single character difference should produce completely different hash
    let hash1 = compute_content_hash("Test content");
    let hash2 = compute_content_hash("Test content."); // Added period

    assert_ne!(hash1, hash2);
}

#[tokio::test]
async fn test_exact_dedup_skips_identical_content() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    // Create a test file
    let test_file = tmp.path().join("test.txt");
    let content = "This is test content for deduplication testing.";
    std::fs::write(&test_file, content)?;

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;

    // First indexing should succeed
    let result1 = rag
        .index_document_with_dedup(&test_file, Some("dedup-test"), SliceMode::Flat)
        .await?;

    assert!(result1.is_indexed(), "First indexing should succeed");

    // Second indexing of SAME content should skip
    let result2 = rag
        .index_document_with_dedup(&test_file, Some("dedup-test"), SliceMode::Flat)
        .await?;

    assert!(
        result2.is_skipped(),
        "Second indexing should be skipped as duplicate"
    );

    // Content hashes should match
    assert_eq!(result1.content_hash(), result2.content_hash());

    Ok(())
}

#[tokio::test]
async fn test_dedup_allows_different_content() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    // Create two test files with different content
    let test_file1 = tmp.path().join("test1.txt");
    let test_file2 = tmp.path().join("test2.txt");
    std::fs::write(&test_file1, "Content of file one.")?;
    std::fs::write(&test_file2, "Content of file two.")?; // Different content

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;

    // Index first file
    let result1 = rag
        .index_document_with_dedup(&test_file1, Some("dedup-test"), SliceMode::Flat)
        .await?;
    assert!(result1.is_indexed());

    // Index second file - should NOT be skipped (different content)
    let result2 = rag
        .index_document_with_dedup(&test_file2, Some("dedup-test"), SliceMode::Flat)
        .await?;
    assert!(result2.is_indexed(), "Different content should be indexed");

    // Hashes should be different
    assert_ne!(result1.content_hash(), result2.content_hash());

    Ok(())
}

#[tokio::test]
async fn test_dedup_different_namespaces() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    // Create a test file
    let test_file = tmp.path().join("test.txt");
    std::fs::write(&test_file, "Same content in different namespaces.")?;

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;

    // Index in first namespace
    let result1 = rag
        .index_document_with_dedup(&test_file, Some("namespace-a"), SliceMode::Flat)
        .await?;
    assert!(result1.is_indexed());

    // Same content in different namespace should also be indexed
    // (dedup is per-namespace, not global)
    let result2 = rag
        .index_document_with_dedup(&test_file, Some("namespace-b"), SliceMode::Flat)
        .await?;
    assert!(
        result2.is_indexed(),
        "Same content in different namespace should be indexed"
    );

    Ok(())
}

#[tokio::test]
async fn test_has_content_hash() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;

    // Create and index a test file
    let test_file = tmp.path().join("test.txt");
    let content = "Content for hash lookup test.";
    std::fs::write(&test_file, content)?;

    let content_hash = compute_content_hash(content);

    // Before indexing, hash should not exist
    let exists_before = rag
        .storage()
        .has_content_hash("hash-test", &content_hash)
        .await?;
    assert!(!exists_before, "Hash should not exist before indexing");

    // Index the file
    let result = rag
        .index_document_with_dedup(&test_file, Some("hash-test"), SliceMode::Flat)
        .await?;
    assert!(result.is_indexed());

    // After indexing, hash should exist
    let exists_after = rag
        .storage()
        .has_content_hash("hash-test", &content_hash)
        .await?;
    assert!(exists_after, "Hash should exist after indexing");

    // Non-existent hash should return false
    let fake_hash = compute_content_hash("non-existent content");
    let fake_exists = rag
        .storage()
        .has_content_hash("hash-test", &fake_hash)
        .await?;
    assert!(!fake_exists, "Non-existent hash should return false");

    Ok(())
}
