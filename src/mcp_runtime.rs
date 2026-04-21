use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

#[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
use crate::{
    ServerConfig,
    embeddings::EmbeddingClient,
    mcp_core::McpCore,
    search::{BM25Index, HybridSearcher},
    security::NamespaceAccessManager,
    storage::StorageManager,
};

/// Build the shared MCP core used by both stdio and HTTP/SSE transports.
pub async fn build_mcp_core(config: ServerConfig) -> Result<Arc<McpCore>> {
    let embedding_client = EmbeddingClient::new(&config.embeddings).await?;
    tracing::info!(
        "Embedding: Connected to {}",
        embedding_client.connected_to()
    );
    let embedding_client = Arc::new(Mutex::new(embedding_client));

    let db_path = shellexpand::tilde(&config.db_path).to_string();
    let storage = Arc::new(StorageManager::new(&db_path).await?);
    let (hybrid_searcher, bm25_writer) = if config.hybrid.mode != crate::search::SearchMode::Vector
    {
        tracing::info!("Hybrid search: mode={:?}", config.hybrid.mode);

        let bm25_reader = Arc::new(BM25Index::new(&config.hybrid.bm25)?);
        let bm25_writer = if config.hybrid.bm25.read_only {
            let mut writer_config = config.hybrid.bm25.clone();
            writer_config.read_only = false;
            Arc::new(BM25Index::new(&writer_config)?)
        } else {
            bm25_reader.clone()
        };

        (
            Some(Arc::new(HybridSearcher::with_bm25_index(
                storage.clone(),
                bm25_reader,
                config.hybrid.clone(),
            ))),
            Some(bm25_writer),
        )
    } else {
        tracing::info!("Hybrid search: disabled (vector-only mode)");
        (None, None)
    };

    let rag = Arc::new(
        crate::rag::RAGPipeline::new_with_bm25(embedding_client.clone(), storage, bm25_writer)
            .await?,
    );

    #[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
    let access_manager = NamespaceAccessManager::new(config.security.clone());
    access_manager.init().await?;
    let access_manager = Arc::new(access_manager);

    Ok(Arc::new(McpCore::new(
        rag,
        hybrid_searcher,
        embedding_client,
        config.max_request_bytes,
        config.allowed_paths,
        access_manager,
    )))
}
