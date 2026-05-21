use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

#[allow(deprecated)]
use crate::{
    embeddings::EmbeddingClient,
    query::{QueryRouter, SearchModeRecommendation},
    rag::{RAGPipeline, SearchOptions, SliceLayer},
    search::{HybridSearcher, SearchMode},
    security::NamespaceAccessManager,
};

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "rmcp-memex";

/// Build a JSON-RPC 2.0 error response.
/// Per JSON-RPC 2.0 spec, omits `id` field when it's null or absent.
pub fn jsonrpc_error(id: Option<&Value>, code: i32, message: impl Into<String>) -> Value {
    let message = message.into();

    match id {
        Some(id) if !id.is_null() => json!({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": id
        }),
        _ => json!({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message}
        }),
    }
}

/// Build a JSON-RPC 2.0 success response.
/// Per JSON-RPC 2.0 spec, omits `id` field when it's null.
pub fn jsonrpc_success(id: &Value, result: Value) -> Value {
    if id.is_null() {
        json!({
            "jsonrpc": "2.0",
            "result": result
        })
    } else {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTransport {
    Stdio,
    HttpSse,
}

impl McpTransport {
    fn health_transport(self) -> Option<&'static str> {
        match self {
            Self::Stdio => None,
            Self::HttpSse => Some("mcp-over-sse"),
        }
    }
}

pub enum McpDispatch {
    Notification,
    Response(Value),
}

impl McpDispatch {
    pub fn into_option(self) -> Option<Value> {
        match self {
            Self::Notification => None,
            Self::Response(response) => Some(response),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpMethod {
    Initialize,
    ToolsList,
    ToolsCall,
}

impl McpMethod {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "initialize" => Some(Self::Initialize),
            "tools/list" => Some(Self::ToolsList),
            "tools/call" => Some(Self::ToolsCall),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpTool {
    Health,
    RagIndex,
    RagIndexText,
    RagSearch,
    MemoryUpsert,
    MemoryGet,
    MemorySearch,
    MemoryDelete,
    MemoryPurgeNamespace,
    NamespaceCreateToken,
    NamespaceRevokeToken,
    NamespaceListProtected,
    NamespaceSecurityStatus,
    Dive,
}

impl McpTool {
    const ALL: [Self; 14] = [
        Self::Health,
        Self::RagIndex,
        Self::RagIndexText,
        Self::RagSearch,
        Self::MemoryUpsert,
        Self::MemoryGet,
        Self::MemorySearch,
        Self::MemoryDelete,
        Self::MemoryPurgeNamespace,
        Self::NamespaceCreateToken,
        Self::NamespaceRevokeToken,
        Self::NamespaceListProtected,
        Self::NamespaceSecurityStatus,
        Self::Dive,
    ];

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "health" => Some(Self::Health),
            "rag_index" => Some(Self::RagIndex),
            "rag_index_text" => Some(Self::RagIndexText),
            "rag_search" => Some(Self::RagSearch),
            "memory_upsert" => Some(Self::MemoryUpsert),
            "memory_get" => Some(Self::MemoryGet),
            "memory_search" => Some(Self::MemorySearch),
            "memory_delete" => Some(Self::MemoryDelete),
            "memory_purge_namespace" => Some(Self::MemoryPurgeNamespace),
            "namespace_create_token" => Some(Self::NamespaceCreateToken),
            "namespace_revoke_token" => Some(Self::NamespaceRevokeToken),
            "namespace_list_protected" => Some(Self::NamespaceListProtected),
            "namespace_security_status" => Some(Self::NamespaceSecurityStatus),
            "dive" => Some(Self::Dive),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::RagIndex => "rag_index",
            Self::RagIndexText => "rag_index_text",
            Self::RagSearch => "rag_search",
            Self::MemoryUpsert => "memory_upsert",
            Self::MemoryGet => "memory_get",
            Self::MemorySearch => "memory_search",
            Self::MemoryDelete => "memory_delete",
            Self::MemoryPurgeNamespace => "memory_purge_namespace",
            Self::NamespaceCreateToken => "namespace_create_token",
            Self::NamespaceRevokeToken => "namespace_revoke_token",
            Self::NamespaceListProtected => "namespace_list_protected",
            Self::NamespaceSecurityStatus => "namespace_security_status",
            Self::Dive => "dive",
        }
    }

    fn definition(self) -> Value {
        match self {
            Self::Health => json!({
                "name": self.name(),
                "description": "Health/status of rmcp-memex server",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }),
            Self::RagIndex => json!({
                "name": self.name(),
                "description": "Index a document for RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "namespace": {"type": "string"}
                    },
                    "required": ["path"]
                }
            }),
            Self::RagIndexText => json!({
                "name": self.name(),
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
            }),
            Self::RagSearch => json!({
                "name": self.name(),
                "description": "Search documents using RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                        "namespace": {"type": "string"},
                        "project": {"type": "string", "description": "Filter to documents whose metadata project/project_id matches this value"},
                        "deep": {"type": "boolean", "default": false, "description": "Include all onion layers instead of only outer summaries"},
                        "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"], "default": "hybrid", "description": "Search mode: vector (semantic), bm25 (keyword), hybrid (both)"},
                        "auto_route": {"type": "boolean", "default": false, "description": "Auto-detect query intent and select optimal search mode. Overrides mode when true."}
                    },
                    "required": ["query"]
                }
            }),
            Self::MemoryUpsert => json!({
                "name": self.name(),
                "description": "Upsert a text chunk into vector memory. If the namespace is protected, provide the access token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                        "token": {"type": "string", "description": "Access token for protected namespaces"}
                    },
                    "required": ["namespace", "id", "text"]
                }
            }),
            Self::MemoryGet => json!({
                "name": self.name(),
                "description": "Get a stored chunk by namespace + id. If the namespace is protected, provide the access token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "id": {"type": "string"},
                        "token": {"type": "string", "description": "Access token for protected namespaces"}
                    },
                    "required": ["namespace", "id"]
                }
            }),
            Self::MemorySearch => json!({
                "name": self.name(),
                "description": "Semantic search within a namespace. If the namespace is protected, provide the access token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                        "project": {"type": "string", "description": "Filter to documents whose metadata project/project_id matches this value"},
                        "deep": {"type": "boolean", "default": false, "description": "Include all onion layers instead of only outer summaries"},
                        "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"], "default": "hybrid", "description": "Search mode: vector (semantic), bm25 (keyword), hybrid (both)"},
                        "auto_route": {"type": "boolean", "default": false, "description": "Auto-detect query intent and select optimal search mode. Overrides mode when true."},
                        "token": {"type": "string", "description": "Access token for protected namespaces"}
                    },
                    "required": ["namespace", "query"]
                }
            }),
            Self::MemoryDelete => json!({
                "name": self.name(),
                "description": "Delete a chunk by namespace + id. If the namespace is protected, provide the access token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "id": {"type": "string"},
                        "token": {"type": "string", "description": "Access token for protected namespaces"}
                    },
                    "required": ["namespace", "id"]
                }
            }),
            Self::MemoryPurgeNamespace => json!({
                "name": self.name(),
                "description": "Delete all chunks in a namespace. If the namespace is protected, provide the access token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "token": {"type": "string", "description": "Access token for protected namespaces"}
                    },
                    "required": ["namespace"]
                }
            }),
            Self::NamespaceCreateToken => json!({
                "name": self.name(),
                "description": "Create an access token for a namespace. Once created, the namespace will require this token for access.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "The namespace to protect with a token"},
                        "description": {"type": "string", "description": "Optional description for the token"}
                    },
                    "required": ["namespace"]
                }
            }),
            Self::NamespaceRevokeToken => json!({
                "name": self.name(),
                "description": "Revoke the access token for a namespace, making it publicly accessible again.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "The namespace to remove token protection from"}
                    },
                    "required": ["namespace"]
                }
            }),
            Self::NamespaceListProtected => json!({
                "name": self.name(),
                "description": "List all namespaces that have token protection enabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }),
            Self::NamespaceSecurityStatus => json!({
                "name": self.name(),
                "description": "Check if namespace security (token-based access control) is enabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }),
            Self::Dive => json!({
                "name": self.name(),
                "description": "Deep exploration with all onion layers. Shows ALL layers (outer/middle/inner/core), both BM25 and vector scores, full metadata, and related chunks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Namespace to search in"},
                        "query": {"type": "string", "description": "Search query text"},
                        "limit": {"type": "integer", "default": 5, "description": "Maximum results per layer"},
                        "verbose": {"type": "boolean", "default": false, "description": "Show full text and metadata"}
                    },
                    "required": ["namespace", "query"]
                }
            }),
        }
    }
}

/// Shared `initialize` result used by every MCP transport.
///
/// rmcp-memex currently exposes a tools-only MCP surface. Do not advertise
/// `resources` here until `resources/list` and related methods are implemented.
pub fn shared_initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {
            "name": SERVER_NAME,
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": {}
        }
    })
}

/// Shared `tools/list` result used by every MCP transport.
pub fn shared_tools_list_result() -> Value {
    let tools: Vec<Value> = McpTool::ALL.into_iter().map(McpTool::definition).collect();
    json!({ "tools": tools })
}

#[derive(Clone)]
#[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
pub struct McpCore {
    rag: Arc<RAGPipeline>,
    hybrid_searcher: Option<Arc<HybridSearcher>>,
    embedding_client: Arc<Mutex<EmbeddingClient>>,
    max_request_bytes: usize,
    allowed_paths: Vec<String>,
    access_manager: Arc<NamespaceAccessManager>,
}

#[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
impl McpCore {
    pub fn new(
        rag: Arc<RAGPipeline>,
        hybrid_searcher: Option<Arc<HybridSearcher>>,
        embedding_client: Arc<Mutex<EmbeddingClient>>,
        max_request_bytes: usize,
        allowed_paths: Vec<String>,
        access_manager: Arc<NamespaceAccessManager>,
    ) -> Self {
        Self {
            rag,
            hybrid_searcher,
            embedding_client,
            max_request_bytes,
            allowed_paths,
            access_manager,
        }
    }

    pub fn rag(&self) -> Arc<RAGPipeline> {
        self.rag.clone()
    }

    /// Access the namespace access manager (if security is enabled).
    /// Returns None when the manager exists but security is disabled.
    #[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; still needed during transition
    pub fn access_manager(&self) -> Option<&NamespaceAccessManager> {
        if self.access_manager.is_enabled() {
            Some(&self.access_manager)
        } else {
            None
        }
    }

    pub fn hybrid_searcher(&self) -> Option<Arc<HybridSearcher>> {
        self.hybrid_searcher.clone()
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embedding_client.lock().await.embed(query).await
    }

    pub async fn handle_request(&self, request: Value, transport: McpTransport) -> Option<Value> {
        self.handle_jsonrpc_request(request, transport)
            .await
            .into_option()
    }

    pub async fn handle_payload(&self, payload: &str, transport: McpTransport) -> Option<Value> {
        let request = match parse_jsonrpc_payload(payload, self.max_request_bytes) {
            Ok(request) => request,
            Err(response) => return Some(response),
        };

        self.handle_request(request, transport).await
    }

    pub async fn handle_jsonrpc_request(
        &self,
        request: Value,
        transport: McpTransport,
    ) -> McpDispatch {
        let method_name = request["method"].as_str().unwrap_or("");

        if method_name.starts_with("notifications/") {
            return McpDispatch::Notification;
        }

        let id = match request.get("id") {
            Some(value) if value.is_string() || value.is_number() => value.clone(),
            _ => {
                return McpDispatch::Response(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing or invalid 'id' field"
                    }
                }));
            }
        };

        let method = match McpMethod::from_name(method_name) {
            Some(method) => method,
            None => {
                return McpDispatch::Response(jsonrpc_error(
                    Some(&id),
                    -32601,
                    format!("Unknown method: {}", method_name),
                ));
            }
        };

        let result = match method {
            McpMethod::Initialize => shared_initialize_result(),
            McpMethod::ToolsList => shared_tools_list_result(),
            McpMethod::ToolsCall => match self.handle_tool_call(&request, &id, transport).await {
                Ok(result) => result,
                Err(response) => return McpDispatch::Response(response),
            },
        };

        McpDispatch::Response(jsonrpc_success(&id, result))
    }

    async fn handle_tool_call(
        &self,
        request: &Value,
        id: &Value,
        transport: McpTransport,
    ) -> std::result::Result<Value, Value> {
        let tool_name = request["params"]["name"].as_str().unwrap_or("");
        let tool = McpTool::from_name(tool_name).ok_or_else(|| {
            jsonrpc_error(Some(id), -32601, format!("Unknown tool: {}", tool_name))
        })?;
        let args = &request["params"]["arguments"];

        match tool {
            McpTool::Health => {
                let mut status = json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "db_path": self.rag.storage_manager().lance_path(),
                    "backend": "mlx",
                    "mlx_server": self.rag.mlx_connected_to(),
                });

                if let Some(transport_name) = transport.health_transport() {
                    status["transport"] = json!(transport_name);
                }

                Ok(text_result_from_json(&status))
            }
            McpTool::RagIndex => {
                let path_str = args["path"].as_str().unwrap_or("");
                let namespace = args["namespace"].as_str();

                let validated_path = validate_path(path_str, &self.allowed_paths)
                    .map_err(|e| jsonrpc_error(Some(id), -32602, e.to_string()))?;

                match self.rag.index_document(&validated_path, namespace).await {
                    Ok(_) => Ok(text_result(format!("Indexed: {}", path_str))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::RagIndexText => {
                let text = args["text"].as_str().unwrap_or("").to_string();
                let namespace = args["namespace"].as_str();
                let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));
                let item_id = args
                    .get("id")
                    .and_then(|value| value.as_str().map(ToOwned::to_owned))
                    .unwrap_or_else(|| Uuid::new_v4().to_string());

                match self
                    .rag
                    .index_text(namespace, item_id.clone(), text, metadata)
                    .await
                {
                    Ok(returned_id) => {
                        Ok(text_result(format!("Indexed text with id {}", returned_id)))
                    }
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::RagSearch => {
                let query = args["query"].as_str().unwrap_or("");
                let limit = requested_limit(args, 10);
                let namespace = args["namespace"].as_str();
                let mode = requested_search_mode(query, args);
                let options = requested_search_options(args);

                if let Some(hybrid_result) = self
                    .try_hybrid_search(
                        query,
                        namespace,
                        limit,
                        (mode, options.clone()),
                        id,
                        SearchShape::Rag,
                    )
                    .await?
                {
                    return Ok(hybrid_result);
                }

                match self
                    .rag
                    .search_with_options(namespace, query, limit, options)
                    .await
                {
                    Ok(results) => Ok(text_result_from_json(&results)),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::MemoryUpsert => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("").to_string();
                let text = args["text"].as_str().unwrap_or("").to_string();
                let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

                match self
                    .rag
                    .memory_upsert(namespace, item_id.clone(), text, metadata)
                    .await
                {
                    Ok(_) => Ok(text_result(format!("Upserted {}", item_id))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::MemoryGet => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("");
                match self.rag.lookup_memory(namespace, item_id).await {
                    Ok(Some(doc)) => Ok(text_result_from_json(&doc)),
                    Ok(None) => Ok(text_result("Not found")),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::MemorySearch => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let query = args["query"].as_str().unwrap_or("");
                let limit = requested_limit(args, 5);
                let mode = requested_search_mode(query, args);
                let options = requested_search_options(args);

                if let Some(hybrid_result) = self
                    .try_hybrid_search(
                        query,
                        Some(namespace),
                        limit,
                        (mode, options.clone()),
                        id,
                        SearchShape::Memory,
                    )
                    .await?
                {
                    return Ok(hybrid_result);
                }

                match self
                    .rag
                    .search_with_options(Some(namespace), query, limit, options)
                    .await
                {
                    Ok(results) => Ok(text_result_from_json(&results)),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::MemoryDelete => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("");
                match self.rag.remove_memory(namespace, item_id).await {
                    Ok(deleted) => Ok(text_result(format!("Deleted {} rows", deleted))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::MemoryPurgeNamespace => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                match self.rag.clear_namespace(namespace).await {
                    Ok(deleted) => Ok(text_result(format!(
                        "Purged namespace '{}', removed {} rows",
                        namespace, deleted
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::NamespaceCreateToken => {
                let namespace = args["namespace"].as_str().unwrap_or("");
                let description = args["description"].as_str().map(ToOwned::to_owned);

                if namespace.is_empty() {
                    return Ok(tool_error_message("Namespace is required"));
                }

                match self
                    .access_manager
                    .create_token(namespace, description)
                    .await
                {
                    Ok(token) => Ok(text_result(format!(
                        "Token created for namespace '{}'. Store this token securely - it won't be shown again!\n\nToken: {}",
                        namespace, token
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::NamespaceRevokeToken => {
                let namespace = args["namespace"].as_str().unwrap_or("");

                if namespace.is_empty() {
                    return Ok(tool_error_message("Namespace is required"));
                }

                match self.access_manager.revoke_token(namespace).await {
                    Ok(true) => Ok(text_result(format!(
                        "Token revoked for namespace '{}'. The namespace is now publicly accessible.",
                        namespace
                    ))),
                    Ok(false) => Ok(text_result(format!(
                        "No token found for namespace '{}'.",
                        namespace
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            McpTool::NamespaceListProtected => {
                let protected = self.access_manager.list_protected_namespaces().await;
                if protected.is_empty() {
                    Ok(text_result(
                        "No namespaces are currently protected with tokens.",
                    ))
                } else {
                    let list: Vec<Value> = protected
                        .iter()
                        .map(|(namespace, created_at, description)| {
                            json!({
                                "namespace": namespace,
                                "created_at": created_at,
                                "description": description
                            })
                        })
                        .collect();
                    Ok(pretty_text_result_from_json(&list))
                }
            }
            McpTool::NamespaceSecurityStatus => {
                let enabled = self.access_manager.is_enabled();
                let protected_count = self.access_manager.list_protected_namespaces().await.len();

                Ok(text_result(format!(
                    "Namespace security: {}\nProtected namespaces: {}\n\nNote: When security is disabled, all namespaces are accessible without tokens.",
                    if enabled { "ENABLED" } else { "DISABLED" },
                    protected_count
                )))
            }
            McpTool::Dive => {
                let namespace = args["namespace"].as_str().unwrap_or("");
                let query = args["query"].as_str().unwrap_or("");
                let limit = args["limit"].as_u64().unwrap_or(5) as usize;
                let verbose = args["verbose"].as_bool().unwrap_or(false);

                if namespace.is_empty() || query.is_empty() {
                    return Err(jsonrpc_error(
                        Some(id),
                        -32602,
                        "namespace and query are required",
                    ));
                }

                let layers = [
                    (Some(SliceLayer::Outer), "outer"),
                    (Some(SliceLayer::Middle), "middle"),
                    (Some(SliceLayer::Inner), "inner"),
                    (Some(SliceLayer::Core), "core"),
                ];

                let mut all_results: Vec<Value> = Vec::new();

                for (layer_filter, layer_name) in &layers {
                    match self
                        .rag
                        .memory_search_with_layer(namespace, query, limit, *layer_filter)
                        .await
                    {
                        Ok(results) => {
                            let layer_results: Vec<Value> = results
                                .iter()
                                .map(|result| {
                                    let mut object = json!({
                                        "id": result.id,
                                        "score": result.score,
                                        "keywords": result.keywords,
                                        "layer": result.layer.map(|layer| layer.name()),
                                        "can_expand": result.can_expand(),
                                        "parent_id": result.parent_id,
                                    });

                                    if verbose {
                                        object["text"] = json!(result.text);
                                        object["metadata"] = result.metadata.clone();
                                        object["children_ids"] = json!(result.children_ids);
                                    } else {
                                        let preview: String =
                                            result.text.chars().take(200).collect();
                                        object["preview"] = json!(preview);
                                    }

                                    object
                                })
                                .collect();

                            all_results.push(json!({
                                "layer": layer_name,
                                "count": results.len(),
                                "results": layer_results
                            }));
                        }
                        Err(e) => {
                            all_results.push(json!({
                                "layer": layer_name,
                                "error": e.to_string()
                            }));
                        }
                    }
                }

                Ok(pretty_text_result_from_json(&json!({
                    "query": query,
                    "namespace": namespace,
                    "limit_per_layer": limit,
                    "verbose": verbose,
                    "layers": all_results
                })))
            }
        }
    }

    async fn try_hybrid_search(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
        search: (SearchMode, SearchOptions),
        id: &Value,
        shape: SearchShape,
    ) -> std::result::Result<Option<Value>, Value> {
        let (mode, options) = search;
        if mode == SearchMode::Vector {
            return Ok(None);
        }

        let Some(hybrid_searcher) = &self.hybrid_searcher else {
            return Ok(None);
        };

        let query_embedding = self
            .embedding_client
            .lock()
            .await
            .embed(query)
            .await
            .map_err(|e| jsonrpc_error(Some(id), -32603, format!("Embedding failed: {}", e)))?;

        let results = hybrid_searcher
            .search(query, query_embedding, namespace, limit, options)
            .await
            .map_err(|e| jsonrpc_error(Some(id), -32603, format!("Hybrid search failed: {}", e)))?;

        let payload: Vec<Value> = match shape {
            SearchShape::Rag => results
                .iter()
                .map(|result| {
                    json!({
                        "id": result.id,
                        "namespace": result.namespace,
                        "document": result.document,
                        "combined_score": result.combined_score,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "metadata": result.metadata,
                        "layer": result.layer.as_ref().map(|layer| format!("{:?}", layer)),
                        "keywords": result.keywords
                    })
                })
                .collect(),
            SearchShape::Memory => results
                .iter()
                .map(|result| {
                    json!({
                        "id": result.id,
                        "namespace": result.namespace,
                        "text": result.document,
                        "score": result.combined_score,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "metadata": result.metadata
                    })
                })
                .collect(),
        };

        Ok(Some(text_result_from_json(&payload)))
    }
}

#[derive(Clone, Copy)]
enum SearchShape {
    Rag,
    Memory,
}

fn requested_search_mode(query: &str, args: &Value) -> SearchMode {
    if args["auto_route"].as_bool().unwrap_or(false) {
        let router = QueryRouter::new();
        let decision = router.route(query);
        match decision.recommended_mode.mode {
            SearchModeRecommendation::Vector => SearchMode::Vector,
            SearchModeRecommendation::Bm25 => SearchMode::Keyword,
            SearchModeRecommendation::Hybrid => SearchMode::Hybrid,
        }
    } else {
        match args["mode"].as_str() {
            Some("vector") => SearchMode::Vector,
            Some("bm25") | Some("keyword") => SearchMode::Keyword,
            _ => SearchMode::Hybrid,
        }
    }
}

fn requested_layer_filter(args: &Value) -> Option<SliceLayer> {
    if args["deep"].as_bool().unwrap_or(false) {
        None
    } else {
        Some(SliceLayer::Outer)
    }
}

fn requested_search_options(args: &Value) -> SearchOptions {
    SearchOptions {
        layer_filter: requested_layer_filter(args),
        project_filter: args["project"]
            .as_str()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    }
}

fn requested_limit(args: &Value, default: usize) -> usize {
    args["k"]
        .as_u64()
        .or_else(|| args["limit"].as_u64())
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn parse_jsonrpc_payload(
    payload: &str,
    max_request_bytes: usize,
) -> std::result::Result<Value, Value> {
    let trimmed = payload.trim();

    if trimmed.len() > max_request_bytes {
        return Err(jsonrpc_error(
            None,
            -32600,
            format!(
                "Request too large: {} bytes (max {})",
                trimmed.len(),
                max_request_bytes
            ),
        ));
    }

    serde_json::from_str(trimmed)
        .map_err(|error| jsonrpc_error(None, -32700, format!("Parse error: {}", error)))
}

fn tool_error(error: impl ToString) -> Value {
    tool_error_message(error.to_string())
}

fn tool_error_message(message: impl Into<String>) -> Value {
    json!({
        "error": {"message": message.into()}
    })
}

fn text_result(text: impl Into<String>) -> Value {
    json!({
        "content": [{"type": "text", "text": text.into()}]
    })
}

fn text_result_from_json<T: serde::Serialize>(value: &T) -> Value {
    text_result(serde_json::to_string(value).unwrap_or_default())
}

fn pretty_text_result_from_json<T: serde::Serialize>(value: &T) -> Value {
    text_result(serde_json::to_string_pretty(value).unwrap_or_default())
}

/// Validates a file path to prevent path traversal attacks.
/// Returns the canonicalized path if valid, or an error if the path is unsafe.
fn validate_path(path_str: &str, allowed_paths: &[String]) -> Result<std::path::PathBuf> {
    if path_str.is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    if path_str.contains("..") || path_str.contains('\0') || path_str.contains('\n') {
        return Err(anyhow!(
            "Path traversal detected: invalid sequences in '{}'",
            path_str
        ));
    }

    let canonical = crate::path_utils::sanitize_existing_path(path_str)?;

    let is_safe = if allowed_paths.is_empty() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .ok();
        let cwd = std::env::current_dir().ok();

        home.as_ref()
            .map(|path| canonical.starts_with(path))
            .unwrap_or(false)
            || cwd
                .as_ref()
                .map(|path| canonical.starts_with(path))
                .unwrap_or(false)
    } else {
        allowed_paths.iter().any(|allowed| {
            let expanded_allowed = shellexpand::tilde(allowed).to_string();
            let allowed_path = Path::new(&expanded_allowed);
            let canonical_allowed = allowed_path
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(&expanded_allowed));

            canonical.starts_with(&canonical_allowed)
        })
    };

    if !is_safe {
        let allowed_info = if allowed_paths.is_empty() {
            "$HOME and current working directory".to_string()
        } else {
            format!("configured paths: {:?}", allowed_paths)
        };

        return Err(anyhow!(
            "Access denied: path '{}' is outside allowed directories ({})",
            path_str,
            allowed_info
        ));
    }

    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::{
        jsonrpc_error, jsonrpc_success, parse_jsonrpc_payload, requested_layer_filter,
        requested_limit, requested_search_options, shared_initialize_result,
        shared_tools_list_result,
    };
    use crate::rag::{SearchOptions, SliceLayer};
    use serde_json::{Value, json};

    #[test]
    fn jsonrpc_error_omits_missing_id() {
        let response = jsonrpc_error(None, -32600, "boom");
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["error"]["code"], -32600);
        assert_eq!(response.get("id"), None);
    }

    #[test]
    fn jsonrpc_success_omits_null_id() {
        let response = jsonrpc_success(&Value::Null, json!({"ok": true}));
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["result"]["ok"], true);
        assert_eq!(response.get("id"), None);
    }

    #[test]
    fn initialize_advertises_only_tools_capability() {
        let response = shared_initialize_result();
        assert_eq!(response["protocolVersion"], "2024-11-05");
        assert_eq!(response["capabilities"], json!({ "tools": {} }));
    }

    #[test]
    fn tool_list_contains_extended_stdio_and_http_surface() {
        let result = shared_tools_list_result();
        let tools = result["tools"]
            .as_array()
            .expect("tools list should be an array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();

        assert!(names.contains(&"rag_index"));
        assert!(names.contains(&"memory_purge_namespace"));
        assert!(names.contains(&"namespace_create_token"));
        assert!(names.contains(&"dive"));
    }

    #[test]
    fn parse_jsonrpc_payload_rejects_oversized_requests() {
        let response = parse_jsonrpc_payload("123456", 5).expect_err("payload should be rejected");
        assert_eq!(response["error"]["code"], -32600);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("Request too large")
        );
    }

    #[test]
    fn parse_jsonrpc_payload_returns_jsonrpc_parse_error() {
        let response = parse_jsonrpc_payload("{", 1024).expect_err("payload should not parse");
        assert_eq!(response["error"]["code"], -32700);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("Parse error")
        );
    }

    #[test]
    fn parse_jsonrpc_payload_accepts_valid_json_with_whitespace() {
        let request = parse_jsonrpc_payload(
            "  {\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}  ",
            1024,
        )
        .expect("payload should parse");

        assert_eq!(request["method"], "initialize");
        assert_eq!(request["id"], 1);
    }

    #[test]
    fn requested_limit_prefers_request_k_over_default() {
        assert_eq!(requested_limit(&json!({"k": 17}), 5), 17);
        assert_eq!(requested_limit(&json!({}), 5), 5);
    }

    #[test]
    fn requested_limit_accepts_limit_alias() {
        assert_eq!(requested_limit(&json!({"limit": 11}), 5), 11);
    }

    #[test]
    fn requested_layer_filter_defaults_to_outer_only() {
        assert_eq!(requested_layer_filter(&json!({})), Some(SliceLayer::Outer));
    }

    #[test]
    fn requested_layer_filter_allows_deep_search() {
        assert_eq!(requested_layer_filter(&json!({"deep": true})), None);
    }

    #[test]
    fn requested_search_options_captures_project_filter() {
        assert_eq!(
            requested_search_options(&json!({"project": "Vista"})),
            SearchOptions {
                layer_filter: Some(SliceLayer::Outer),
                project_filter: Some("Vista".to_string()),
            }
        );
    }
}
