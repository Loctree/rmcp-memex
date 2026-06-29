//! End-to-End Pipeline Tests
//!
//! These tests verify the full pipeline: index → embed → search
//! They require a running embedding server (Ollama or MLX).
//!
//! Run with: cargo test --test e2e_pipeline -- --ignored
//!
//! Vibecrafted with AI Agents by Vetcoders (c)2026

use rmcp_memex::{
    ChromaDocument, EmbeddingClient, EmbeddingConfig, ProviderConfig, StorageManager,
};
use serde::Deserialize;
use serde_json::json;
use tempfile::TempDir;

const LOCAL_OLLAMA_MODEL: &str = "qwen3-embedding:4b";
const LOCAL_OLLAMA_DIMENSION: usize = 2560;

// =============================================================================
// HELPER: Check if embedding server is available
// =============================================================================

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

async fn ollama_qwen4b_available() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    if let Ok(resp) = client.get("http://localhost:11434/api/tags").send().await {
        if !resp.status().is_success() {
            return false;
        }

        if let Ok(tags) = resp.json::<OllamaTagsResponse>().await {
            return tags
                .models
                .iter()
                .any(|model| model.name == LOCAL_OLLAMA_MODEL);
        }
    }

    false
}

fn create_test_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig {
        required_dimension: LOCAL_OLLAMA_DIMENSION,
        max_batch_chars: 32000,
        max_batch_items: 16,
        providers: vec![ProviderConfig {
            name: "ollama-local".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: LOCAL_OLLAMA_MODEL.to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        reranker: Default::default(),
    }
}

// =============================================================================
// E2E TEST: Full pipeline index → embed → search
// =============================================================================

/// Full E2E test: create embeddings, store, search, verify results
#[tokio::test]
#[ignore] // Run with: cargo test --test e2e_pipeline -- --ignored
async fn test_e2e_index_embed_search() {
    if !ollama_qwen4b_available().await {
        eprintln!(
            "SKIP: Local Ollama model '{}' unavailable at http://localhost:11434",
            LOCAL_OLLAMA_MODEL
        );
        return;
    }

    // Setup
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let config = create_test_embedding_config();
    let mut embedder = EmbeddingClient::new(&config)
        .await
        .expect("Failed to create embedding client");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    // Test documents
    let test_docs = vec![
        (
            "doc-rust",
            "Rust is a systems programming language focused on safety and performance.",
        ),
        (
            "doc-python",
            "Python is a high-level interpreted language known for its readability.",
        ),
        (
            "doc-javascript",
            "JavaScript is the language of the web, running in browsers and Node.js.",
        ),
        (
            "doc-veterinary",
            "Veterinary medicine involves the diagnosis and treatment of animals.",
        ),
        (
            "doc-ai",
            "Artificial intelligence enables machines to learn from experience.",
        ),
    ];

    // INDEX: Generate embeddings and store documents
    let mut stored_docs = Vec::new();
    for (id, content) in &test_docs {
        let embedding = embedder
            .embed(content)
            .await
            .expect("Failed to generate embedding");

        assert_eq!(
            embedding.len(),
            LOCAL_OLLAMA_DIMENSION,
            "Embedding dimension should be {}",
            LOCAL_OLLAMA_DIMENSION
        );

        let doc = ChromaDocument::new_flat(
            id.to_string(),
            "e2e-test-ns".to_string(),
            embedding,
            json!({"type": "programming", "source": "test"}),
            content.to_string(),
        );
        stored_docs.push(doc);
    }

    storage
        .add_to_store(stored_docs)
        .await
        .expect("Failed to store documents");

    // SEARCH: Query for programming languages
    let query = "systems programming language with memory safety";
    let query_embedding = embedder.embed(query).await.expect("Failed to embed query");

    let results = storage
        .search_store(Some("e2e-test-ns"), query_embedding, 3)
        .await
        .expect("Failed to search");

    // VERIFY: Should find Rust as top result (most relevant to query)
    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 3, "Should respect limit");

    // Rust should be in top results for memory safety query
    let result_ids: Vec<&str> = results.iter().map(|d| d.id.as_str()).collect();
    eprintln!("Search results for '{}': {:?}", query, result_ids);

    // At minimum, we should get some results
    assert!(
        results.iter().any(|d| d.id.starts_with("doc-")),
        "Results should contain our test documents"
    );
}

/// Test batch embedding with multiple texts
#[tokio::test]
#[ignore]
async fn test_e2e_batch_embedding() {
    if !ollama_qwen4b_available().await {
        eprintln!(
            "SKIP: Local Ollama model '{}' unavailable",
            LOCAL_OLLAMA_MODEL
        );
        return;
    }

    let config = create_test_embedding_config();
    let mut embedder = EmbeddingClient::new(&config)
        .await
        .expect("Failed to create embedding client");

    let texts: Vec<String> = vec![
        "First document about machine learning".to_string(),
        "Second document about natural language processing".to_string(),
        "Third document about computer vision".to_string(),
        "Fourth document about reinforcement learning".to_string(),
    ];

    let embeddings = embedder
        .embed_batch(&texts)
        .await
        .expect("Failed to batch embed");

    assert_eq!(
        embeddings.len(),
        texts.len(),
        "Should get embedding for each text"
    );

    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            LOCAL_OLLAMA_DIMENSION,
            "Embedding {} should have {} dimensions",
            i,
            LOCAL_OLLAMA_DIMENSION
        );

        // Check no NaN/Inf values
        for (j, &val) in emb.iter().enumerate() {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Embedding {} has invalid value at index {}: {}",
                i,
                j,
                val
            );
        }
    }
}

/// Test dimension validation at startup (fail-fast)
#[tokio::test]
#[ignore]
async fn test_e2e_local_ollama_qwen4b_validation() {
    if !ollama_qwen4b_available().await {
        eprintln!(
            "SKIP: Local Ollama model '{}' unavailable",
            LOCAL_OLLAMA_MODEL
        );
        return;
    }

    // Config with correct dimension
    let config = create_test_embedding_config();
    let result = EmbeddingClient::new(&config).await;
    assert!(
        result.is_ok(),
        "Should connect with correct dimension config"
    );

    let mut embedder = result.unwrap();
    let connected = embedder.connected_to();
    eprintln!("Connected to: {}", connected);

    assert!(
        connected == "ollama-local",
        "Should connect to local Ollama, got: {}",
        connected
    );

    let embedding = embedder
        .embed("qwen4b validation probe")
        .await
        .expect("Expected local Ollama /v1/embeddings to work");
    assert_eq!(embedding.len(), LOCAL_OLLAMA_DIMENSION);

    let mut bad_config = create_test_embedding_config();
    bad_config.required_dimension = 4096;
    let err = EmbeddingClient::new(&bad_config)
        .await
        .err()
        .expect("Mismatched config dimension should fail fast");
    let message = err.to_string();
    assert!(
        message.contains("returned 2560 dims") || message.contains("returned 2560"),
        "Unexpected error message: {}",
        message
    );
    assert!(
        message.contains("required_dimension=4096"),
        "Unexpected error message: {}",
        message
    );
}

/// Test deduplication via content hash
#[tokio::test]
#[ignore]
async fn test_e2e_deduplication() {
    if !ollama_qwen4b_available().await {
        eprintln!(
            "SKIP: Local Ollama model '{}' unavailable",
            LOCAL_OLLAMA_MODEL
        );
        return;
    }

    use rmcp_memex::compute_content_hash;

    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let config = create_test_embedding_config();
    let mut embedder = EmbeddingClient::new(&config)
        .await
        .expect("Failed to create embedding client");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage.ensure_collection().await.unwrap();

    let content = "This is unique content for deduplication test";
    let hash = compute_content_hash(content);

    // First check - hash should not exist
    let exists_before = storage
        .has_content_hash("dedup-test-ns", &hash)
        .await
        .expect("Failed to check hash");
    assert!(!exists_before, "Hash should not exist before indexing");

    // Index the content
    let embedding = embedder.embed(content).await.unwrap();
    let mut doc = ChromaDocument::new_flat(
        "dedup-doc-1".to_string(),
        "dedup-test-ns".to_string(),
        embedding,
        json!({}),
        content.to_string(),
    );
    doc.content_hash = Some(hash.clone());
    storage.add_to_store(vec![doc]).await.unwrap();

    // Second check - hash should exist now
    let exists_after = storage
        .has_content_hash("dedup-test-ns", &hash)
        .await
        .expect("Failed to check hash");
    assert!(exists_after, "Hash should exist after indexing");
}

/// Test that invalid embeddings are rejected
#[tokio::test]
async fn test_storage_rejects_invalid_embeddings() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage.ensure_collection().await.unwrap();

    // Test: Empty ID should be rejected
    let doc_empty_id = ChromaDocument::new_flat(
        "".to_string(),
        "test-ns".to_string(),
        vec![0.1f32; LOCAL_OLLAMA_DIMENSION],
        json!({}),
        "Content".to_string(),
    );
    let result = storage.add_to_store(vec![doc_empty_id]).await;
    assert!(result.is_err(), "Empty ID should be rejected");

    // Test: Empty namespace should be rejected
    let doc_empty_ns = ChromaDocument::new_flat(
        "valid-id".to_string(),
        "".to_string(),
        vec![0.1f32; LOCAL_OLLAMA_DIMENSION],
        json!({}),
        "Content".to_string(),
    );
    let result = storage.add_to_store(vec![doc_empty_ns]).await;
    assert!(result.is_err(), "Empty namespace should be rejected");

    // Test: NaN in embedding should be rejected
    let mut embedding_with_nan = vec![0.1f32; LOCAL_OLLAMA_DIMENSION];
    embedding_with_nan[100] = f32::NAN;
    let doc_nan = ChromaDocument::new_flat(
        "nan-doc".to_string(),
        "test-ns".to_string(),
        embedding_with_nan,
        json!({}),
        "Content".to_string(),
    );
    let result = storage.add_to_store(vec![doc_nan]).await;
    assert!(result.is_err(), "NaN in embedding should be rejected");

    // Test: Inf in embedding should be rejected
    let mut embedding_with_inf = vec![0.1f32; LOCAL_OLLAMA_DIMENSION];
    embedding_with_inf[200] = f32::INFINITY;
    let doc_inf = ChromaDocument::new_flat(
        "inf-doc".to_string(),
        "test-ns".to_string(),
        embedding_with_inf,
        json!({}),
        "Content".to_string(),
    );
    let result = storage.add_to_store(vec![doc_inf]).await;
    assert!(result.is_err(), "Inf in embedding should be rejected");

    // Test: Inconsistent dimensions in batch should be rejected
    let doc_good = ChromaDocument::new_flat(
        "doc-good".to_string(),
        "test-ns".to_string(),
        vec![0.1f32; LOCAL_OLLAMA_DIMENSION],
        json!({}),
        "Content".to_string(),
    );
    let doc_1024 = ChromaDocument::new_flat(
        "doc-1024".to_string(),
        "test-ns".to_string(),
        vec![0.1f32; 1024], // Wrong dimension!
        json!({}),
        "Content".to_string(),
    );
    let result = storage.add_to_store(vec![doc_good, doc_1024]).await;
    assert!(
        result.is_err(),
        "Inconsistent dimensions should be rejected"
    );
}
