//! HTTP/SSE server for rmcp-memex
//!
//! Provides HTTP endpoints for agents that can't hold LanceDB lock directly.
//! All database access goes through the single server instance.
//!
//! Uses RAGPipeline (same as MCPServer) for consistency and full feature support:
//! - Multi-namespace (each agent can have own namespace)
//! - Onion slices (expand/drill-down in SSE)
//! - Full indexing pipeline with dedup
//!
//! Endpoints:
//! - GET  /health           - Health check
//! - POST /search           - Search documents
//! - GET  /sse/search       - SSE streaming search
//! - POST /upsert           - Upsert document (memory_upsert)
//! - POST /index            - Index text with full pipeline
//! - GET  /expand/:ns/:id   - Expand onion slice (get children)
//! - GET  /parent/:ns/:id   - Get parent slice (drill up)
//! - DELETE /ns/:namespace  - Purge namespace
//!
//! MCP-over-SSE endpoints (for Claude Code compatibility):
//! - GET  /mcp/             - SSE stream for MCP messages (sends endpoint event)
//! - POST /mcp/messages/    - JSON-RPC POST endpoint with session_id
//!
//! Created by M&K (c)2025 The LibraxisAI Team

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{RwLock, broadcast};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, warn};

use crate::rag::{RAGPipeline, SearchResult, SliceLayer};

/// MCP session for SSE connections
pub struct McpSession {
    /// Session ID
    pub id: String,
    /// Channel to send responses back to SSE stream
    pub tx: broadcast::Sender<serde_json::Value>,
    /// Created timestamp
    pub created: std::time::Instant,
}

/// MCP session manager
pub struct McpSessionManager {
    sessions: RwLock<HashMap<String, Arc<McpSession>>>,
}

impl McpSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Create new session and return session ID
    pub async fn create_session(&self) -> (String, broadcast::Receiver<serde_json::Value>) {
        let id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = broadcast::channel(64);
        let session = Arc::new(McpSession {
            id: id.clone(),
            tx,
            created: std::time::Instant::now(),
        });
        self.sessions.write().await.insert(id.clone(), session);
        (id, rx)
    }

    /// Get session by ID
    pub async fn get_session(&self, id: &str) -> Option<Arc<McpSession>> {
        self.sessions.read().await.get(id).cloned()
    }

    /// Remove session
    pub async fn remove_session(&self, id: &str) {
        self.sessions.write().await.remove(id);
    }

    /// Cleanup old sessions (older than 1 hour)
    pub async fn cleanup_old_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|_, s| s.created.elapsed() < Duration::from_secs(3600));
    }
}

impl Default for McpSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared state for HTTP handlers - uses RAGPipeline like MCPServer
#[derive(Clone)]
pub struct HttpState {
    pub rag: Arc<RAGPipeline>,
    /// MCP session manager for SSE transport
    pub mcp_sessions: Arc<McpSessionManager>,
    /// Base URL for MCP messages endpoint (set at startup)
    pub mcp_base_url: Arc<RwLock<String>>,
}

/// Search request body
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Optional layer filter for onion slices
    #[serde(default)]
    pub layer: Option<u8>,
}

fn default_limit() -> usize {
    10
}

/// Search result for JSON response
#[derive(Debug, Serialize)]
pub struct SearchResultJson {
    pub id: String,
    pub namespace: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Can expand to children (has children_ids)
    pub can_expand: bool,
    /// Can drill up to parent (has parent_id)
    pub can_drill_up: bool,
}

impl From<SearchResult> for SearchResultJson {
    fn from(r: SearchResult) -> Self {
        let can_expand = r.can_expand();
        let can_drill_up = r.can_drill_up();
        Self {
            id: r.id,
            namespace: r.namespace,
            text: r.text,
            score: r.score,
            metadata: r.metadata,
            layer: r.layer.map(|l| l.name().to_string()),
            parent_id: r.parent_id,
            children_ids: r.children_ids,
            keywords: r.keywords,
            can_expand,
            can_drill_up,
        }
    }
}

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultJson>,
    pub query: String,
    pub namespace: Option<String>,
    pub elapsed_ms: u64,
    pub count: usize,
}

/// Upsert request body (memory_upsert)
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub namespace: String,
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Index text request (full pipeline)
#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    pub namespace: String,
    pub content: String,
    /// Slice mode: "flat", "outer", "deep" (default: "flat")
    #[serde(default = "default_slice_mode")]
    pub slice_mode: String,
}

fn default_slice_mode() -> String {
    "flat".to_string()
}

/// SSE search query params
#[derive(Debug, Deserialize)]
pub struct SseSearchParams {
    pub query: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Cross-search request - search across all namespaces
#[derive(Debug, Deserialize)]
pub struct CrossSearchRequest {
    pub query: String,
    /// Limit per namespace (default: 5)
    #[serde(default = "default_cross_limit")]
    pub limit: usize,
    /// Total limit across all namespaces (default: 20)
    #[serde(default = "default_total_limit")]
    pub total_limit: usize,
    /// Search mode: "vector", "keyword"/"bm25", "hybrid" (default: hybrid)
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_cross_limit() -> usize {
    5
}

fn default_total_limit() -> usize {
    20
}

fn default_mode() -> String {
    "hybrid".to_string()
}

/// Cross-search query params for GET endpoint
#[derive(Debug, Deserialize)]
pub struct CrossSearchParams {
    #[serde(rename = "q")]
    pub query: String,
    #[serde(default = "default_cross_limit")]
    pub limit: usize,
    #[serde(default = "default_total_limit")]
    pub total_limit: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
}

/// Cross-search response
#[derive(Debug, Serialize)]
pub struct CrossSearchResponse {
    pub results: Vec<SearchResultJson>,
    pub query: String,
    pub mode: String,
    pub namespaces_searched: usize,
    pub total_results: usize,
    pub elapsed_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub db_path: String,
    pub embedding_provider: String,
}

/// Create the HTTP router
pub fn create_router(state: HttpState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_handler))
        .route("/refresh", post(refresh_handler))
        .route("/search", post(search_handler))
        .route("/sse/search", get(sse_search_handler))
        .route("/cross-search", get(cross_search_handler))
        .route("/sse/cross-search", get(sse_cross_search_handler))
        .route("/upsert", post(upsert_handler))
        .route("/index", post(index_handler))
        .route("/expand/{ns}/{id}", get(expand_handler))
        .route("/parent/{ns}/{id}", get(parent_handler))
        .route("/get/{ns}/{id}", get(get_handler))
        .route("/delete/{ns}/{id}", post(delete_handler))
        .route("/ns/{namespace}", delete(purge_namespace_handler))
        // MCP-over-SSE endpoints for Claude Code compatibility
        .route("/mcp/", get(mcp_sse_handler))
        .route("/mcp/messages/", post(mcp_messages_handler))
        // Also support /sse/ path for FastMCP compatibility
        .route("/sse/", get(mcp_sse_handler))
        .route("/messages/", post(mcp_messages_handler))
        .layer(cors)
        .with_state(state)
}

/// Health check endpoint
async fn health_handler(State(state): State<HttpState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        db_path: state.rag.storage().lance_path().to_string(),
        embedding_provider: state.rag.mlx_connected_to(),
    })
}

/// Refresh endpoint - clears LanceDB cache to see new data from other processes
async fn refresh_handler(
    State(state): State<HttpState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    state.rag.refresh().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Refresh failed: {}", e),
        )
    })?;

    Ok(Json(serde_json::json!({
        "status": "refreshed",
        "message": "LanceDB cache cleared - next query will see fresh data"
    })))
}

/// Search endpoint (POST /search)
async fn search_handler(
    State(state): State<HttpState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let results = if let Some(layer_u8) = req.layer {
        // Search with layer filter
        let layer = SliceLayer::from_u8(layer_u8);
        state
            .rag
            .memory_search_with_layer(
                req.namespace.as_deref().unwrap_or("default"),
                &req.query,
                req.limit,
                layer,
            )
            .await
    } else {
        // Regular search
        state
            .rag
            .memory_search(
                req.namespace.as_deref().unwrap_or("default"),
                &req.query,
                req.limit,
            )
            .await
    }
    .map_err(|e| {
        error!("Search error: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let count = results.len();
    let search_results: Vec<SearchResultJson> = results.into_iter().map(Into::into).collect();

    Ok(Json(SearchResponse {
        results: search_results,
        query: req.query,
        namespace: req.namespace,
        elapsed_ms: start.elapsed().as_millis() as u64,
        count,
    }))
}

/// SSE streaming search endpoint (GET /sse/search?query=...&namespace=...&limit=...)
async fn sse_search_handler(
    State(state): State<HttpState>,
    Query(params): Query<SseSearchParams>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        // Send start event
        yield Ok(Event::default()
            .event("start")
            .data(serde_json::json!({
                "query": params.query,
                "namespace": params.namespace,
                "limit": params.limit
            }).to_string()));

        let namespace = params.namespace.as_deref().unwrap_or("default");

        match state.rag.memory_search(namespace, &params.query, params.limit).await {
            Ok(results) => {
                let total = results.len();

                for (i, r) in results.into_iter().enumerate() {
                    let result: SearchResultJson = r.into();

                    if let Ok(json) = serde_json::to_string(&result) {
                        yield Ok(Event::default()
                            .event("result")
                            .id(i.to_string())
                            .data(json));
                    }

                    // Small delay for streaming effect
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }

                yield Ok(Event::default()
                    .event("done")
                    .data(serde_json::json!({
                        "status": "complete",
                        "total": total
                    }).to_string()));
            }
            Err(e) => {
                yield Ok(Event::default()
                    .event("error")
                    .data(serde_json::json!({"error": e.to_string()}).to_string()));
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// Cross-search endpoint (GET /cross-search?q=...&limit=...&total_limit=...&mode=...)
/// Searches across ALL namespaces, merges results by score
async fn cross_search_handler(
    State(state): State<HttpState>,
    Query(params): Query<CrossSearchParams>,
) -> Result<Json<CrossSearchResponse>, (StatusCode, String)> {
    use std::collections::HashSet;

    let start = std::time::Instant::now();

    // Get all namespaces by searching with zero embedding
    let zero_embedding = vec![0.0_f32; 4096]; // Max dimension
    let all_docs = state
        .rag
        .storage()
        .search_store(None, zero_embedding, 10000)
        .await
        .map_err(|e| {
            error!("Cross-search namespace lookup error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    let mut namespace_set: HashSet<String> = HashSet::new();
    for doc in &all_docs {
        namespace_set.insert(doc.namespace.clone());
    }

    let namespaces: Vec<String> = namespace_set.into_iter().collect();
    let namespaces_count = namespaces.len();

    if namespaces.is_empty() {
        return Ok(Json(CrossSearchResponse {
            results: vec![],
            query: params.query,
            mode: params.mode,
            namespaces_searched: 0,
            total_results: 0,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }));
    }

    // Search each namespace
    let mut all_results: Vec<(SearchResultJson, f32)> = Vec::new();

    for ns in &namespaces {
        match state
            .rag
            .memory_search(ns, &params.query, params.limit)
            .await
        {
            Ok(results) => {
                for r in results {
                    let score = r.score;
                    all_results.push((r.into(), score));
                }
            }
            Err(e) => {
                // Log but continue - don't fail entire search for one namespace
                error!("Cross-search error in namespace '{}': {}", ns, e);
            }
        }
    }

    // Sort by score descending
    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to total_limit
    all_results.truncate(params.total_limit);

    let results: Vec<SearchResultJson> = all_results.into_iter().map(|(r, _)| r).collect();
    let total_results = results.len();

    Ok(Json(CrossSearchResponse {
        results,
        query: params.query,
        mode: params.mode,
        namespaces_searched: namespaces_count,
        total_results,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

/// SSE streaming cross-search endpoint (GET /sse/cross-search?q=...&limit=...&total_limit=...)
/// Streams results as they come from each namespace
async fn sse_cross_search_handler(
    State(state): State<HttpState>,
    Query(params): Query<CrossSearchParams>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    use std::collections::HashSet;

    let stream = async_stream::stream! {
        // Send start event
        yield Ok(Event::default()
            .event("start")
            .data(serde_json::json!({
                "query": params.query,
                "limit_per_ns": params.limit,
                "total_limit": params.total_limit,
                "mode": params.mode
            }).to_string()));

        // Get all namespaces
        let zero_embedding = vec![0.0_f32; 4096];
        let all_docs = match state.rag.storage().search_store(None, zero_embedding, 10000).await {
            Ok(docs) => docs,
            Err(e) => {
                yield Ok(Event::default()
                    .event("error")
                    .data(serde_json::json!({"error": e.to_string()}).to_string()));
                return;
            }
        };

        let mut namespace_set: HashSet<String> = HashSet::new();
        for doc in &all_docs {
            namespace_set.insert(doc.namespace.clone());
        }

        let namespaces: Vec<String> = namespace_set.into_iter().collect();

        // Send namespace info
        yield Ok(Event::default()
            .event("namespaces")
            .data(serde_json::json!({
                "count": namespaces.len(),
                "namespaces": namespaces
            }).to_string()));

        // Collect all results with scores for final ranking
        let mut all_results: Vec<(SearchResultJson, f32, String)> = Vec::new();

        // Search each namespace and stream intermediate results
        for ns in &namespaces {
            yield Ok(Event::default()
                .event("searching")
                .data(serde_json::json!({"namespace": ns}).to_string()));

            match state.rag.memory_search(ns, &params.query, params.limit).await {
                Ok(results) => {
                    let ns_count = results.len();
                    for r in results {
                        let score = r.score;
                        let result: SearchResultJson = r.into();
                        all_results.push((result, score, ns.clone()));
                    }

                    yield Ok(Event::default()
                        .event("namespace_done")
                        .data(serde_json::json!({
                            "namespace": ns,
                            "results_found": ns_count
                        }).to_string()));
                }
                Err(e) => {
                    yield Ok(Event::default()
                        .event("namespace_error")
                        .data(serde_json::json!({
                            "namespace": ns,
                            "error": e.to_string()
                        }).to_string()));
                }
            }

            // Small delay between namespaces
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Sort all results by score descending
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate and stream final ranked results
        all_results.truncate(params.total_limit);

        for (i, (result, _score, _ns)) in all_results.iter().enumerate() {
            if let Ok(json) = serde_json::to_string(&result) {
                yield Ok(Event::default()
                    .event("result")
                    .id(i.to_string())
                    .data(json));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        yield Ok(Event::default()
            .event("done")
            .data(serde_json::json!({
                "status": "complete",
                "total_results": all_results.len(),
                "namespaces_searched": namespaces.len()
            }).to_string()));
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// Upsert document endpoint (POST /upsert) - uses memory_upsert
async fn upsert_handler(
    State(state): State<HttpState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let metadata = req.metadata.unwrap_or(serde_json::json!({}));

    state
        .rag
        .memory_upsert(
            &req.namespace,
            req.id.clone(),
            req.content.clone(),
            metadata,
        )
        .await
        .map_err(|e| {
            error!("Upsert error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "id": req.id,
        "namespace": req.namespace
    })))
}

/// Index text with full pipeline (POST /index)
async fn index_handler(
    State(state): State<HttpState>,
    Json(req): Json<IndexRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    use crate::rag::SliceMode;

    let mode = match req.slice_mode.as_str() {
        "onion" => SliceMode::Onion,
        "onion_fast" | "fast" => SliceMode::OnionFast,
        _ => SliceMode::Flat,
    };

    // Generate ID from content hash
    let id = format!(
        "idx_{}",
        uuid::Uuid::new_v4()
            .to_string()
            .split('-')
            .next()
            .unwrap_or("000")
    );

    let result_id = state
        .rag
        .index_text_with_mode(
            Some(&req.namespace),
            id,
            req.content.clone(),
            serde_json::json!({}),
            mode,
        )
        .await
        .map_err(|e| {
            error!("Index error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    Ok(Json(serde_json::json!({
        "status": "indexed",
        "namespace": req.namespace,
        "id": result_id,
        "slice_mode": req.slice_mode
    })))
}

/// Expand onion slice - get children (GET /expand/:ns/:id)
async fn expand_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let children = state.rag.expand_result(&ns, &id).await.map_err(|e| {
        error!("Expand error: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let results: Vec<SearchResultJson> = children.into_iter().map(Into::into).collect();

    Ok(Json(serde_json::json!({
        "parent_id": id,
        "namespace": ns,
        "children": results,
        "count": results.len()
    })))
}

/// Get parent slice - drill up (GET /parent/:ns/:id)
async fn parent_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.get_parent_result(&ns, &id).await {
        Ok(Some(parent)) => {
            let result: SearchResultJson = parent.into();
            Ok(Json(serde_json::json!({
                "child_id": id,
                "namespace": ns,
                "parent": result
            })))
        }
        Ok(None) => Err((StatusCode::NOT_FOUND, format!("No parent for '{}'", id))),
        Err(e) => {
            error!("Parent error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Get document by namespace and ID (GET /get/:ns/:id)
async fn get_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.memory_get(&ns, &id).await {
        Ok(Some(r)) => {
            let result: SearchResultJson = r.into();
            Ok(Json(serde_json::json!(result)))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            format!("Document '{}' not found in '{}'", id, ns),
        )),
        Err(e) => {
            error!("Get error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Delete document (POST /delete/:ns/:id)
async fn delete_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.memory_delete(&ns, &id).await {
        Ok(deleted) => Ok(Json(serde_json::json!({
            "status": if deleted > 0 { "deleted" } else { "not_found" },
            "id": id,
            "namespace": ns
        }))),
        Err(e) => {
            error!("Delete error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Purge entire namespace (DELETE /ns/:namespace)
async fn purge_namespace_handler(
    State(state): State<HttpState>,
    Path(namespace): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.purge_namespace(&namespace).await {
        Ok(deleted) => Ok(Json(serde_json::json!({
            "status": "purged",
            "namespace": namespace,
            "deleted_count": deleted
        }))),
        Err(e) => {
            error!("Purge error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

// ============================================================================
// MCP-over-SSE Transport Handlers
// ============================================================================

/// Query params for MCP messages endpoint
#[derive(Debug, Deserialize)]
pub struct McpMessagesParams {
    pub session_id: Option<String>,
}

/// MCP SSE endpoint - GET /sse/ or /mcp/
/// Creates a new session and sends the endpoint URL for messages
async fn mcp_sse_handler(
    State(state): State<HttpState>,
    headers: axum::http::HeaderMap,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    // Create a new session
    let (session_id, mut rx) = state.mcp_sessions.create_session().await;

    // Use Host header from request to build endpoint URL (enables remote access)
    let base_url = if let Some(host) = headers.get(axum::http::header::HOST) {
        if let Ok(host_str) = host.to_str() {
            format!("http://{}", host_str)
        } else {
            state.mcp_base_url.read().await.clone()
        }
    } else {
        state.mcp_base_url.read().await.clone()
    };

    info!("MCP SSE: New session {} (base_url: {})", session_id, base_url);

    let stream = async_stream::stream! {
        // First event: tell client where to POST messages (FastMCP/MCP SSE protocol)
        let endpoint_url = format!("{}/messages/?session_id={}", base_url, session_id);
        yield Ok(Event::default()
            .event("endpoint")
            .data(endpoint_url));

        // Keep connection alive and forward responses from the session
        loop {
            tokio::select! {
                // Receive responses from session channel
                result = rx.recv() => {
                    match result {
                        Ok(response) => {
                            if let Ok(json_str) = serde_json::to_string(&response) {
                                yield Ok(Event::default()
                                    .event("message")
                                    .data(json_str));
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            debug!("MCP SSE: Session {} channel closed", session_id);
                            break;
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("MCP SSE: Session {} lagged {} messages", session_id, n);
                        }
                    }
                }
                // Keep-alive ping every 30 seconds
                _ = tokio::time::sleep(Duration::from_secs(30)) => {
                    // SSE keepalive is handled by axum's KeepAlive
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// MCP Messages endpoint - POST /messages/?session_id=xxx
/// Receives JSON-RPC requests and sends responses via SSE stream
/// Returns 202 Accepted - actual response delivered via SSE
async fn mcp_messages_handler(
    State(state): State<HttpState>,
    Query(params): Query<McpMessagesParams>,
    body: String,
) -> Result<StatusCode, (StatusCode, String)> {
    let session_id = params.session_id.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "session_id is required".to_string(),
        )
    })?;

    // Get the session
    let session = state
        .mcp_sessions
        .get_session(&session_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Session {} not found", session_id),
            )
        })?;

    // Parse the JSON-RPC request
    let request: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)))?;

    debug!("MCP: session={} method={}", session_id, request["method"]);

    // Handle the request (inline MCP protocol handling)
    let response = handle_mcp_request(&state.rag, request).await;

    // Only send response for requests (not notifications)
    if let Some(response) = response
        && let Err(e) = session.tx.send(response)
    {
        warn!(
            "MCP: Failed to send response to session {}: {}",
            session_id, e
        );
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to send response".to_string(),
        ));
    }

    // Return 202 Accepted - actual response (if any) goes via SSE stream
    Ok(StatusCode::ACCEPTED)
}

/// Handle MCP JSON-RPC request
/// Implements core MCP protocol methods using RAGPipeline
/// Returns None for notifications (no response needed), Some for requests
async fn handle_mcp_request(
    rag: &Arc<RAGPipeline>,
    request: serde_json::Value,
) -> Option<serde_json::Value> {
    let method = request["method"].as_str().unwrap_or("");
    let id = request.get("id").cloned();

    // Notifications start with "notifications/" - no response per JSON-RPC spec
    if method.starts_with("notifications/") {
        debug!("MCP: notification '{}' - no response", method);
        return None;
    }

    // Requests must have an id (string or number)
    let id = match id {
        Some(v) if v.is_string() || v.is_number() => v,
        _ => {
            warn!("MCP: request '{}' missing valid id", method);
            return Some(json!({
                "jsonrpc": "2.0",
                "id": serde_json::Value::Null,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing or invalid 'id' field"
                }
            }));
        }
    };

    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "rmcp-memex",
                "version": env!("CARGO_PKG_VERSION")
            },
            "capabilities": {
                "tools": {}
            }
        }),

        "tools/list" => json!({
            "tools": [
                {
                    "name": "health",
                    "description": "Health/status of rmcp-memex server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "rag_index_text",
                    "description": "Index raw text for RAG/memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "id": {"type": "string"},
                            "namespace": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "rag_search",
                    "description": "Search documents using RAG",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 10},
                            "namespace": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "memory_upsert",
                    "description": "Upsert a text chunk into vector memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["namespace", "id", "text"]
                    }
                },
                {
                    "name": "memory_search",
                    "description": "Semantic search within a namespace",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 5}
                        },
                        "required": ["namespace", "query"]
                    }
                },
                {
                    "name": "memory_get",
                    "description": "Get a stored chunk by namespace + id",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "id": {"type": "string"}
                        },
                        "required": ["namespace", "id"]
                    }
                },
                {
                    "name": "memory_delete",
                    "description": "Delete a chunk by namespace + id",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "id": {"type": "string"}
                        },
                        "required": ["namespace", "id"]
                    }
                }
            ]
        }),

        "tools/call" => {
            let tool_name = request["params"]["name"].as_str().unwrap_or("");
            let args = &request["params"]["arguments"];

            match tool_name {
                "health" => {
                    let status = json!({
                        "version": env!("CARGO_PKG_VERSION"),
                        "db_path": rag.storage().lance_path(),
                        "backend": "mlx",
                        "transport": "mcp-over-sse"
                    });
                    json!({
                        "content": [{"type": "text", "text": serde_json::to_string(&status).unwrap_or_default()}]
                    })
                }
                "rag_index_text" => {
                    let text = args["text"].as_str().unwrap_or("").to_string();
                    let namespace = args["namespace"].as_str();
                    let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));
                    let id = args
                        .get("id")
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                    match rag.index_text(namespace, id.clone(), text, metadata).await {
                        Ok(returned_id) => json!({
                            "content": [{"type": "text", "text": format!("Indexed text with id {}", returned_id)}]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                "rag_search" => {
                    let query = args["query"].as_str().unwrap_or("");
                    let k = args["k"].as_u64().unwrap_or(10) as usize;
                    let namespace = args["namespace"].as_str();

                    match rag.search_inner(namespace, query, k).await {
                        Ok(results) => json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string(&results).unwrap_or_default()
                            }]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                "memory_upsert" => {
                    let namespace = args["namespace"].as_str().unwrap_or("default");
                    let id_str = args["id"].as_str().unwrap_or("").to_string();
                    let text = args["text"].as_str().unwrap_or("").to_string();
                    let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

                    match rag
                        .memory_upsert(namespace, id_str.clone(), text, metadata)
                        .await
                    {
                        Ok(_) => json!({
                            "content": [{"type": "text", "text": format!("Upserted {}", id_str)}]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                "memory_search" => {
                    let namespace = args["namespace"].as_str().unwrap_or("default");
                    let query = args["query"].as_str().unwrap_or("");
                    let k = args["k"].as_u64().unwrap_or(5) as usize;

                    match rag.memory_search(namespace, query, k).await {
                        Ok(results) => json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string(&results).unwrap_or_default()
                            }]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                "memory_get" => {
                    let namespace = args["namespace"].as_str().unwrap_or("default");
                    let id_str = args["id"].as_str().unwrap_or("");
                    match rag.memory_get(namespace, id_str).await {
                        Ok(Some(doc)) => json!({
                            "content": [{"type": "text", "text": serde_json::to_string(&doc).unwrap_or_default()}]
                        }),
                        Ok(None) => json!({
                            "content": [{"type": "text", "text": "Not found"}]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                "memory_delete" => {
                    let namespace = args["namespace"].as_str().unwrap_or("default");
                    let id_str = args["id"].as_str().unwrap_or("");
                    match rag.memory_delete(namespace, id_str).await {
                        Ok(deleted) => json!({
                            "content": [{"type": "text", "text": format!("Deleted {} rows", deleted)}]
                        }),
                        Err(e) => json!({
                            "error": {"message": e.to_string()}
                        }),
                    }
                }
                _ => {
                    return Some(json!({
                        "jsonrpc": "2.0",
                        "error": {"code": -32601, "message": format!("Unknown tool: {}", tool_name)},
                        "id": id
                    }));
                }
            }
        }

        _ => {
            return Some(json!({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": format!("Unknown method: {}", method)},
                "id": id
            }));
        }
    };

    Some(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }))
}

/// Start the HTTP server with shared RAGPipeline
pub async fn start_server(rag: Arc<RAGPipeline>, port: u16) -> anyhow::Result<()> {
    // Fallback base_url - actual URL is derived from Host header in mcp_sse_handler
    let base_url = format!("http://localhost:{}", port);
    let state = HttpState {
        rag,
        mcp_sessions: Arc::new(McpSessionManager::new()),
        mcp_base_url: Arc::new(RwLock::new(base_url.clone())),
    };
    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("HTTP/SSE server starting on http://{}", addr);
    info!("  REST endpoints: /search, /sse/search, /upsert, /expand, /parent");
    info!("  MCP-SSE endpoints: /sse/, /messages/");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_request_defaults() {
        let json = r#"{"query": "test"}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
        assert!(req.namespace.is_none());
        assert!(req.layer.is_none());
    }

    #[test]
    fn test_index_request_defaults() {
        let json = r#"{"namespace": "test", "content": "hello"}"#;
        let req: IndexRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.slice_mode, "flat");
    }
}
