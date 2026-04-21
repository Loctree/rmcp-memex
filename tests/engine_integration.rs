//! Integration tests for MemexEngine
//!
//! These tests focus on the storage layer using temp directories.
//! Tests that require embeddings (Ollama) are skipped if unavailable.

use axum::{Json, Router, extract::State, routing::post};
use rmcp_memex::{
    BatchResult, ChromaDocument, EmbeddingConfig, MemexConfig, MemexEngine, MetaFilter,
    ProviderConfig, StorageManager, StoreItem,
};
use serde::Deserialize;
use serde_json::json;
use tempfile::TempDir;
use tokio::{net::TcpListener, task::JoinHandle};

#[derive(Clone)]
struct MockEmbeddingState {
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct MockEmbeddingRequest {
    input: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct MockEmbeddingResponse {
    data: Vec<MockEmbeddingData>,
}

#[derive(Debug, serde::Serialize)]
struct MockEmbeddingData {
    embedding: Vec<f32>,
}

async fn mock_embeddings(
    State(state): State<MockEmbeddingState>,
    Json(request): Json<MockEmbeddingRequest>,
) -> Json<MockEmbeddingResponse> {
    let data = request
        .input
        .into_iter()
        .map(|_| MockEmbeddingData {
            embedding: vec![0.25_f32; state.dimension],
        })
        .collect();

    Json(MockEmbeddingResponse { data })
}

async fn start_mock_embedding_server(dimension: usize) -> (String, JoinHandle<()>) {
    let app = Router::new()
        .route("/v1/embeddings", post(mock_embeddings))
        .with_state(MockEmbeddingState { dimension });

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind mock embedding server");
    let address = listener
        .local_addr()
        .expect("Failed to get mock embedding server address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock embedding server failed");
    });

    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    (format!("http://{}", address), handle)
}

// =============================================================================
// STORAGE MANAGER TESTS (No embeddings required)
// =============================================================================

/// Test StorageManager initialization with temp directory
#[tokio::test]
async fn test_storage_manager_init() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");
}

/// Test MemexConfig construction and builders
#[test]
fn test_memex_config_full_chain() {
    let config = MemexConfig::new("test-app", "my-namespace")
        .with_dimension(512)
        .with_db_path("/tmp/custom/path");

    assert_eq!(config.app_name, "test-app");
    assert_eq!(config.namespace, "my-namespace");
    assert_eq!(config.dimension, 512);
    assert_eq!(config.effective_db_path(), "/tmp/custom/path");
}

/// Test MetaFilter with all field types
#[test]
fn test_meta_filter_comprehensive() {
    // Test patient filter
    let patient_filter = MetaFilter::for_patient("P-12345");
    assert_eq!(patient_filter.patient_id, Some("P-12345".to_string()));
    assert!(patient_filter.visit_id.is_none());

    // Test visit filter
    let visit_filter = MetaFilter::for_visit("V-67890");
    assert_eq!(visit_filter.visit_id, Some("V-67890".to_string()));
    assert!(visit_filter.patient_id.is_none());

    // Test custom filter chain
    let custom_filter = MetaFilter::default()
        .with_custom("species", "canine")
        .with_custom("breed", "labrador");

    assert_eq!(custom_filter.custom.len(), 2);
}

/// Test MetaFilter matching logic with edge cases
#[test]
fn test_meta_filter_matching_edge_cases() {
    // Test null values in metadata
    let filter = MetaFilter::for_patient("P-123");
    let null_patient = json!({"patient_id": null});
    assert!(!filter.matches(&null_patient));

    // Test numeric values (should not match string filter)
    let numeric_id = json!({"patient_id": 123});
    assert!(!filter.matches(&numeric_id));

    // Test array value (should not match)
    let array_id = json!({"patient_id": ["P-123"]});
    assert!(!filter.matches(&array_id));

    // Test nested object (should not match)
    let nested = json!({"patient_id": {"id": "P-123"}});
    assert!(!filter.matches(&nested));
}

/// Test StoreItem construction
#[test]
fn test_store_item_construction() {
    let item1 = StoreItem::new("id-1", "Some text content");
    assert_eq!(item1.id, "id-1");
    assert_eq!(item1.text, "Some text content");
    assert!(item1.metadata.is_object());

    let item2 = StoreItem::new("id-2", "More content").with_metadata(json!({
        "patient_id": "P-456",
        "doc_type": "soap_note",
        "tags": ["urgent", "follow-up"]
    }));
    assert_eq!(item2.metadata["patient_id"], "P-456");
    assert!(item2.metadata["tags"].is_array());
}

/// Test BatchResult structure
#[test]
fn test_batch_result_structure() {
    let result = BatchResult {
        success_count: 95,
        failure_count: 5,
        failed_ids: vec![
            "doc-10".into(),
            "doc-20".into(),
            "doc-30".into(),
            "doc-40".into(),
            "doc-50".into(),
        ],
    };

    assert_eq!(result.success_count, 95);
    assert_eq!(result.failure_count, 5);
    assert_eq!(result.failed_ids.len(), 5);

    // Verify we can iterate over failed IDs
    for id in &result.failed_ids {
        assert!(id.starts_with("doc-"));
    }
}

/// Test StorageManager document operations without embeddings
#[tokio::test]
async fn test_storage_document_operations() {
    use rmcp_memex::ChromaDocument;

    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    // Create a test document with dummy embedding
    let dummy_embedding = vec![0.1f32; 4096];
    let doc = ChromaDocument::new_flat(
        "test-doc-1".to_string(),
        "test-namespace".to_string(),
        dummy_embedding.clone(),
        json!({"type": "test", "patient_id": "P-123"}),
        "Test document content".to_string(),
    );

    // Add document
    storage
        .add_to_store(vec![doc])
        .await
        .expect("Failed to add document");

    // Retrieve document
    let retrieved = storage
        .get_document("test-namespace", "test-doc-1")
        .await
        .expect("Failed to get document");

    assert!(retrieved.is_some());
    let doc = retrieved.unwrap();
    assert_eq!(doc.id, "test-doc-1");
    assert_eq!(doc.namespace, "test-namespace");
    assert_eq!(doc.document, "Test document content");

    // Delete document
    let deleted = storage
        .delete_document("test-namespace", "test-doc-1")
        .await
        .expect("Failed to delete document");

    assert!(deleted > 0);

    // Verify it's gone
    let after_delete = storage
        .get_document("test-namespace", "test-doc-1")
        .await
        .expect("Failed to get document");

    assert!(after_delete.is_none());
}

/// Test search with dummy embeddings (verifies storage layer)
#[tokio::test]
async fn test_storage_search_operations() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    // Add multiple documents with varying embeddings
    let docs: Vec<ChromaDocument> = (0..5)
        .map(|i| {
            let mut embedding = vec![0.0f32; 4096];
            embedding[0] = i as f32 * 0.1;
            ChromaDocument::new_flat(
                format!("doc-{}", i),
                "search-ns".to_string(),
                embedding,
                json!({"index": i}),
                format!("Document number {}", i),
            )
        })
        .collect();

    storage
        .add_to_store(docs)
        .await
        .expect("Failed to add documents");

    // Search with a query embedding
    let query_embedding = vec![0.0f32; 4096];
    let results = storage
        .search_store(Some("search-ns"), query_embedding, 3)
        .await
        .expect("Failed to search");

    assert!(!results.is_empty());
    assert!(results.len() <= 3);
}

/// Test namespace isolation
#[tokio::test]
async fn test_namespace_isolation() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    let embedding = vec![0.5f32; 4096];

    // Add to namespace A
    let doc_a = ChromaDocument::new_flat(
        "shared-id".to_string(),
        "namespace-a".to_string(),
        embedding.clone(),
        json!({"ns": "a"}),
        "Content in namespace A".to_string(),
    );
    storage.add_to_store(vec![doc_a]).await.unwrap();

    // Add to namespace B with same ID
    let doc_b = ChromaDocument::new_flat(
        "shared-id".to_string(),
        "namespace-b".to_string(),
        embedding.clone(),
        json!({"ns": "b"}),
        "Content in namespace B".to_string(),
    );
    storage.add_to_store(vec![doc_b]).await.unwrap();

    // Get from namespace A
    let from_a = storage
        .get_document("namespace-a", "shared-id")
        .await
        .unwrap();
    assert!(from_a.is_some());
    assert_eq!(from_a.unwrap().document, "Content in namespace A");

    // Get from namespace B
    let from_b = storage
        .get_document("namespace-b", "shared-id")
        .await
        .unwrap();
    assert!(from_b.is_some());
    assert_eq!(from_b.unwrap().document, "Content in namespace B");

    // Delete from A should not affect B
    storage
        .delete_document("namespace-a", "shared-id")
        .await
        .unwrap();

    let after_delete_a = storage
        .get_document("namespace-a", "shared-id")
        .await
        .unwrap();
    assert!(after_delete_a.is_none());

    let still_in_b = storage
        .get_document("namespace-b", "shared-id")
        .await
        .unwrap();
    assert!(still_in_b.is_some());
}

/// Test content hash deduplication
#[tokio::test]
async fn test_content_hash_storage() {
    use rmcp_memex::compute_content_hash;

    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    let content = "This is some test content for hashing";
    let hash = compute_content_hash(content);

    // Initially hash should not exist
    let exists_before = storage
        .has_content_hash("hash-test-ns", &hash)
        .await
        .expect("Failed to check hash");
    assert!(!exists_before);

    // After adding document with this content, hash should exist
    use rmcp_memex::ChromaDocument;
    let embedding = vec![0.1f32; 4096];
    let mut doc = ChromaDocument::new_flat(
        "hash-doc".to_string(),
        "hash-test-ns".to_string(),
        embedding,
        json!({}),
        content.to_string(),
    );
    doc.content_hash = Some(hash.clone());

    storage.add_to_store(vec![doc]).await.unwrap();

    let exists_after = storage
        .has_content_hash("hash-test-ns", &hash)
        .await
        .expect("Failed to check hash");
    assert!(exists_after);
}

#[tokio::test]
async fn test_delete_by_filter_scans_past_first_page() {
    const DIMENSION: usize = 8;
    let (base_url, server_handle) = start_mock_embedding_server(DIMENSION).await;

    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let embedding_config = EmbeddingConfig {
        required_dimension: DIMENSION,
        max_batch_chars: 32_000,
        max_batch_items: 16,
        providers: vec![ProviderConfig {
            name: "mock".to_string(),
            base_url,
            model: "mock-embedder".to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        reranker: Default::default(),
    };

    let mut config = MemexConfig::new("delete-filter-test", "filter-ns")
        .with_dimension(DIMENSION)
        .with_db_path(db_path.to_str().unwrap())
        .with_embedding_config(embedding_config);
    config.enable_hybrid = false;

    let engine = MemexEngine::new(config)
        .await
        .expect("Failed to create engine");
    let storage = engine.storage();

    let docs: Vec<ChromaDocument> = (0..1105)
        .map(|idx| {
            let doc_type = if idx >= 1000 { "target" } else { "keep" };
            ChromaDocument::new_flat(
                format!("doc-{idx:04}"),
                "filter-ns".to_string(),
                vec![0.5_f32; DIMENSION],
                json!({ "doc_type": doc_type, "idx": idx }),
                format!("Document {idx}"),
            )
        })
        .collect();

    storage
        .add_to_store(docs)
        .await
        .expect("Failed to seed documents");

    let deleted = engine
        .delete_by_filter(MetaFilter {
            doc_type: Some("target".to_string()),
            ..Default::default()
        })
        .await
        .expect("delete_by_filter should succeed");

    assert_eq!(
        deleted, 105,
        "all target docs past the first page should delete"
    );

    let remaining = storage
        .all_documents(Some("filter-ns"), 2000)
        .await
        .expect("Failed to read remaining documents");
    let remaining_target = remaining
        .iter()
        .filter(|doc| doc.metadata["doc_type"] == "target")
        .count();
    let remaining_keep = remaining
        .iter()
        .filter(|doc| doc.metadata["doc_type"] == "keep")
        .count();

    assert_eq!(remaining_target, 0, "target docs should be fully removed");
    assert_eq!(remaining_keep, 1000, "non-matching docs should remain");

    let deleted_again = engine
        .delete_by_filter(MetaFilter {
            doc_type: Some("target".to_string()),
            ..Default::default()
        })
        .await
        .expect("idempotent delete_by_filter should succeed");
    assert_eq!(deleted_again, 0, "second delete should find nothing");

    server_handle.abort();
}

// =============================================================================
// BATCH OPERATIONS TESTS
// =============================================================================

/// Test adding many documents in batch
#[tokio::test]
async fn test_batch_add_many_documents() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let db_path = tmp.path().join("lancedb");

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("Failed to create storage");

    storage
        .ensure_collection()
        .await
        .expect("Failed to ensure collection");

    // Create 100 documents
    let docs: Vec<ChromaDocument> = (0..100)
        .map(|i| {
            let mut embedding = vec![0.0f32; 4096];
            embedding[i % 4096] = 1.0;
            ChromaDocument::new_flat(
                format!("batch-doc-{}", i),
                "batch-ns".to_string(),
                embedding,
                json!({"batch_index": i, "type": "batch-test"}),
                format!("Batch document content number {}", i),
            )
        })
        .collect();

    // Add all at once
    storage
        .add_to_store(docs)
        .await
        .expect("Failed to batch add documents");

    // Verify a few documents exist
    for i in [0, 25, 50, 75, 99] {
        let doc = storage
            .get_document("batch-ns", &format!("batch-doc-{}", i))
            .await
            .expect("Failed to get document");
        assert!(doc.is_some(), "Document batch-doc-{} should exist", i);
    }
}

// =============================================================================
// SERIALIZATION TESTS
// =============================================================================

/// Test MemexConfig JSON serialization round-trip
#[test]
fn test_memex_config_json_roundtrip() {
    use rmcp_memex::search::BM25Config;

    let original = MemexConfig::new("serialization-test", "test-ns")
        .with_dimension(768)
        .with_db_path("/custom/db/path")
        .with_bm25(BM25Config::default());

    let json = serde_json::to_string_pretty(&original).expect("Failed to serialize");
    let deserialized: MemexConfig = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.app_name, original.app_name);
    assert_eq!(deserialized.namespace, original.namespace);
    assert_eq!(deserialized.dimension, original.dimension);
    assert_eq!(
        deserialized.effective_db_path(),
        original.effective_db_path()
    );
    assert_eq!(deserialized.enable_bm25, original.enable_bm25);
}

/// Test MetaFilter TOML serialization (for config files)
#[test]
fn test_meta_filter_toml_roundtrip() {
    let original = MetaFilter {
        patient_id: Some("P-TOML-TEST".to_string()),
        visit_id: Some("V-123".to_string()),
        doc_type: Some("soap_note".to_string()),
        date_from: Some("2024-01-01".to_string()),
        date_to: Some("2024-12-31".to_string()),
        custom: vec![("key".to_string(), "value".to_string())],
    };

    let toml = toml::to_string_pretty(&original).expect("Failed to serialize to TOML");
    let deserialized: MetaFilter = toml::from_str(&toml).expect("Failed to deserialize from TOML");

    assert_eq!(deserialized.patient_id, original.patient_id);
    assert_eq!(deserialized.visit_id, original.visit_id);
    assert_eq!(deserialized.doc_type, original.doc_type);
    assert_eq!(deserialized.custom, original.custom);
}

/// Test StoreItem with complex metadata
#[test]
fn test_store_item_complex_metadata() {
    let complex_metadata = json!({
        "patient_id": "P-COMPLEX",
        "tags": ["urgent", "follow-up", "lab-results"],
        "vitals": {
            "heart_rate": 75,
            "temperature": 38.5,
            "weight_kg": 4.2
        },
        "medications": [
            {"name": "Amoxicillin", "dosage_mg": 250},
            {"name": "Meloxicam", "dosage_mg": 0.5}
        ],
        "is_emergency": false
    });

    let item =
        StoreItem::new("complex-doc", "Patient medical record").with_metadata(complex_metadata);

    // Verify nested access works
    assert_eq!(item.metadata["patient_id"], "P-COMPLEX");
    assert_eq!(item.metadata["tags"][0], "urgent");
    assert_eq!(item.metadata["vitals"]["heart_rate"], 75);
    assert_eq!(item.metadata["medications"][0]["name"], "Amoxicillin");
}
