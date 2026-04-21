use crate::{
    BM25Config, BM25Index, ChromaDocument, CrossStoreRecoveryBatch, CrossStoreRecoveryStatus,
    EmbeddingConfig, MLXBridge, ProviderConfig, SearchOptions, SliceLayer, SliceMode,
    compute_content_hash, inspect_cross_store_recovery, rag::RAGPipeline,
    repair_cross_store_recovery, storage::StorageManager,
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
        .lookup_memory("testns", "doc1")
        .await?
        .ok_or_else(|| anyhow!("doc missing"))?;
    assert_eq!(fetched.text, "Ala ma kota");
    assert_eq!(fetched.namespace, "testns");

    // Semantic search within namespace
    let results = rag.search_memory("testns", "kota", 1).await?;
    assert!(!results.is_empty(), "expected at least one search result");
    assert_eq!(results[0].namespace, "testns");

    Ok(())
}

#[tokio::test]
async fn rag_pipeline_syncs_bm25_writes() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");
    let bm25_path = tmp.path().join(".bm25");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let bm25 = Arc::new(BM25Index::new(
        &BM25Config::default().with_path(bm25_path.to_string_lossy().into_owned()),
    )?);
    let rag = RAGPipeline::new_with_bm25(mlx, storage, Some(bm25.clone())).await?;

    rag.index_text_with_mode(
        Some("testns"),
        "doc1".to_string(),
        "Ala ma kota".to_string(),
        json!({"lang": "pl"}),
        SliceMode::Flat,
    )
    .await?;

    let results = bm25.search("kota", Some("testns"), 10)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");

    Ok(())
}

#[tokio::test]
async fn rag_pipeline_rolls_back_lance_write_when_bm25_write_fails() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");
    let bm25_path = tmp.path().join(".bm25");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let bm25 = Arc::new(BM25Index::new(
        &BM25Config::default()
            .with_path(bm25_path.to_string_lossy().into_owned())
            .with_read_only(true),
    )?);
    let rag = RAGPipeline::new_with_bm25(mlx, storage.clone(), Some(bm25)).await?;

    let err = rag
        .index_text_with_mode(
            Some("rollback-ns"),
            "doc1".to_string(),
            "This batch should never survive a failed BM25 write because rollback keeps LanceDB truthful."
                .to_string(),
            json!({"kind": "rollback-guard"}),
            SliceMode::Flat,
        )
        .await
        .expect_err("read-only BM25 should reject the write path");

    assert!(
        err.to_string().contains("read-only mode"),
        "unexpected error: {err}"
    );
    assert!(
        storage
            .get_all_in_namespace("rollback-ns")
            .await?
            .is_empty(),
        "LanceDB rows should be rolled back when BM25 indexing fails"
    );

    Ok(())
}

#[tokio::test]
async fn repair_cross_store_replays_bm25_for_pending_lance_batch() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");
    let bm25_path = tmp.path().join(".bm25");

    let storage = StorageManager::new_lance_only(&db_path.to_string_lossy()).await?;
    storage.ensure_collection().await?;

    let bm25 =
        BM25Index::new(&BM25Config::default().with_path(bm25_path.to_string_lossy().into_owned()))?;

    let text = "Recoverable divergence batch should be replayed into BM25 after Lance survives.";
    let document = ChromaDocument::new_flat_with_hash(
        "doc1".to_string(),
        "repair-ns".to_string(),
        vec![0.1, 0.2],
        json!({"kind": "repairable"}),
        text.to_string(),
        compute_content_hash(text),
    );
    let batch = CrossStoreRecoveryBatch::from_documents(std::slice::from_ref(&document));

    storage.persist_cross_store_recovery_batch(&batch)?;
    storage.add_to_store(vec![document]).await?;

    let inspect = inspect_cross_store_recovery(&storage, &bm25, Some("repair-ns")).await?;
    assert_eq!(inspect.pending_batches, 1);
    assert_eq!(inspect.divergent_batches, 1);
    assert_eq!(inspect.documents_missing_bm25, 1);

    let repaired = repair_cross_store_recovery(&storage, &bm25, Some("repair-ns")).await?;
    assert_eq!(repaired.batches_repaired, 1);
    assert_eq!(repaired.repaired_documents, 1);
    assert_eq!(repaired.cleared_batches, 1);
    assert!(storage.list_cross_store_recovery_batches()?.is_empty());

    let results = bm25.search("recoverable", Some("repair-ns"), 10)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");

    Ok(())
}

#[tokio::test]
async fn repair_cross_store_clears_rolled_back_batches_as_skipped() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");
    let bm25_path = tmp.path().join(".bm25");

    let storage = StorageManager::new_lance_only(&db_path.to_string_lossy()).await?;
    storage.ensure_collection().await?;

    let bm25 =
        BM25Index::new(&BM25Config::default().with_path(bm25_path.to_string_lossy().into_owned()))?;

    let text = "This batch was rolled back before BM25 could keep it.";
    let document = ChromaDocument::new_flat_with_hash(
        "doc1".to_string(),
        "rollback-ledger-ns".to_string(),
        vec![0.3, 0.4],
        json!({"kind": "rolled-back"}),
        text.to_string(),
        compute_content_hash(text),
    );
    let mut batch = CrossStoreRecoveryBatch::from_documents(std::slice::from_ref(&document));
    batch.status = CrossStoreRecoveryStatus::RolledBack;
    batch.last_error =
        Some("BM25 write failed after Lance persist; lance_rollback_failures=0".to_string());
    storage.persist_cross_store_recovery_batch(&batch)?;

    let inspect = inspect_cross_store_recovery(&storage, &bm25, Some("rollback-ledger-ns")).await?;
    assert_eq!(inspect.pending_batches, 1);
    assert_eq!(inspect.rolled_back_batches, 1);
    assert_eq!(inspect.documents_missing_lance, 1);

    let repaired = repair_cross_store_recovery(&storage, &bm25, Some("rollback-ledger-ns")).await?;
    assert_eq!(repaired.rolled_back_batches, 1);
    assert_eq!(repaired.skipped_documents, 1);
    assert_eq!(repaired.cleared_batches, 1);
    assert!(storage.list_cross_store_recovery_batches()?.is_empty());

    Ok(())
}

#[tokio::test]
async fn onion_text_index_overrides_stale_slice_mode_metadata() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage).await?;

    rag.index_text_with_mode(
        Some("onion-test"),
        "doc1".to_string(),
        "This is a longer document about onions and embeddings. It should create layered slices for metadata verification."
            .to_string(),
        json!({"slice_mode": "flat"}),
        SliceMode::OnionFast,
    )
    .await?;

    let docs = rag
        .storage_manager()
        .get_all_in_namespace("onion-test")
        .await?;
    assert!(!docs.is_empty(), "expected onion slices to be stored");
    assert!(
        docs.iter()
            .all(|doc| doc.metadata["slice_mode"] == "onion-fast"),
        "all stored slices should carry the actual onion slice_mode"
    );

    Ok(())
}

#[test]
fn search_options_default_to_outer_layer() {
    assert_eq!(
        SearchOptions::default().layer_filter,
        Some(SliceLayer::Outer)
    );
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

    assert!(result1.was_indexed(), "First indexing should succeed");

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
    assert!(result1.was_indexed());

    // Index second file - should NOT be skipped (different content)
    let result2 = rag
        .index_document_with_dedup(&test_file2, Some("dedup-test"), SliceMode::Flat)
        .await?;
    assert!(result2.was_indexed(), "Different content should be indexed");

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
    assert!(result1.was_indexed());

    // Same content in different namespace should also be indexed
    // (dedup is per-namespace, not global)
    let result2 = rag
        .index_document_with_dedup(&test_file, Some("namespace-b"), SliceMode::Flat)
        .await?;
    assert!(
        result2.was_indexed(),
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
        .storage_manager()
        .has_content_hash("hash-test", &content_hash)
        .await?;
    assert!(!exists_before, "Hash should not exist before indexing");

    // Index the file
    let result = rag
        .index_document_with_dedup(&test_file, Some("hash-test"), SliceMode::Flat)
        .await?;
    assert!(result.was_indexed());

    // After indexing, hash should exist
    let exists_after = rag
        .storage_manager()
        .has_content_hash("hash-test", &content_hash)
        .await?;
    assert!(exists_after, "Hash should exist after indexing");

    // Non-existent hash should return false
    let fake_hash = compute_content_hash("non-existent content");
    let fake_exists = rag
        .storage_manager()
        .has_content_hash("hash-test", &fake_hash)
        .await?;
    assert!(!fake_exists, "Non-existent hash should return false");

    Ok(())
}

#[tokio::test]
async fn memory_upsert_replaces_onion_family_by_original_id() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;
    let namespace = "memory-upsert-onion";
    let initial_text = "This is a long document about indexing and embeddings. It exists to force onion slicing during the first upsert. \
The content is intentionally verbose and repetitive so the outer and core layers both exist and can later be replaced cleanly. \
Project Vista keeps appearing here to provide a stable metadata anchor for search filters.";
    let updated_text = "This is an updated long document about project search filters, namespace cache invalidation, and release hardening. \
It should fully replace the previous onion family instead of appending stale slices. \
The text stays comfortably above the slicing threshold so onion-fast still writes both outer and core documents.";

    rag.memory_upsert(
        namespace,
        "doc-1".to_string(),
        initial_text.to_string(),
        json!({"slice_mode": "onion-fast", "project": "Vista"}),
    )
    .await?;

    let initial_docs = rag
        .storage_manager()
        .get_all_in_namespace(namespace)
        .await?;
    assert_eq!(
        initial_docs.len(),
        2,
        "onion-fast should create outer + core"
    );
    assert!(
        initial_docs
            .iter()
            .all(|doc| doc.metadata["original_id"] == "doc-1")
    );

    let fetched = rag
        .lookup_memory(namespace, "doc-1")
        .await?
        .ok_or_else(|| anyhow!("expected lookup by original id to resolve onion family"))?;
    assert_eq!(fetched.layer, Some(SliceLayer::Outer));
    assert_eq!(fetched.metadata["original_id"], "doc-1");

    rag.memory_upsert(
        namespace,
        "doc-1".to_string(),
        updated_text.to_string(),
        json!({"slice_mode": "onion-fast", "project": "Vista"}),
    )
    .await?;

    let updated_docs = rag
        .storage_manager()
        .get_all_in_namespace(namespace)
        .await?;
    assert_eq!(
        updated_docs.len(),
        2,
        "upsert should replace the onion family instead of duplicating it"
    );
    assert!(
        updated_docs
            .iter()
            .all(|doc| doc.metadata["original_id"] == "doc-1")
    );
    assert!(
        updated_docs
            .iter()
            .any(|doc| doc.document.contains("namespace cache invalidation")),
        "updated content should be present in the stored family"
    );

    let deleted = rag.remove_memory(namespace, "doc-1").await?;
    assert_eq!(
        deleted, 2,
        "remove_memory should delete the whole onion family"
    );
    assert!(
        rag.storage_manager()
            .get_all_in_namespace(namespace)
            .await?
            .is_empty()
    );

    Ok(())
}

#[tokio::test]
async fn flat_memory_upsert_keeps_original_id_for_chunked_family_operations() -> Result<()> {
    let mlx = require_mlx!(try_mlx_bridge().await);

    let tmp = tempfile::tempdir()?;
    let db_path = tmp.path().join(".lancedb");

    let storage = Arc::new(StorageManager::new_lance_only(&db_path.to_string_lossy()).await?);
    storage.ensure_collection().await?;

    let rag = RAGPipeline::new(mlx, storage.clone()).await?;
    let namespace = "memory-upsert-flat";
    let original_id = "../customer-notes/../../alpha";
    let text = "Operator notes about cache invalidation, path hardening, and release readiness. "
        .repeat(80);

    rag.memory_upsert(
        namespace,
        original_id.to_string(),
        text,
        json!({"project": "Pathless"}),
    )
    .await?;

    let docs = rag
        .storage_manager()
        .get_all_in_namespace(namespace)
        .await?;
    assert!(
        docs.len() > 1,
        "flat memory upsert should chunk this long document into a family"
    );
    assert!(
        docs.iter()
            .all(|doc| doc.metadata["original_id"] == original_id),
        "every flat family chunk should remember the caller's original id"
    );
    assert!(
        docs.iter()
            .all(|doc| doc.id == original_id || doc.id.contains("::chunk::")),
        "chunk ids should stay in the memory-id domain instead of pretending to be file paths"
    );

    let fetched = rag
        .lookup_memory(namespace, original_id)
        .await?
        .ok_or_else(|| anyhow!("expected lookup by original id to resolve flat family"))?;
    assert_eq!(fetched.metadata["original_id"], original_id);
    assert_eq!(
        fetched.metadata["chunk_index"], 0,
        "lookup should deterministically return the first flat chunk"
    );

    let deleted = rag.remove_memory(namespace, original_id).await?;
    assert_eq!(
        deleted,
        docs.len(),
        "remove_memory should delete the whole flat family by original id"
    );
    assert!(
        rag.storage_manager()
            .get_all_in_namespace(namespace)
            .await?
            .is_empty()
    );

    Ok(())
}
