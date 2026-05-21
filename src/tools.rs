//! In-process helper functions for memory-oriented operations.
//!
//! These helpers are convenient for Rust callers embedding `rmcp-memex`
//! directly. The authoritative MCP tool contract exposed over stdio and
//! HTTP/SSE comes from `tool_definitions()`, which mirrors the shared runtime
//! surface instead of maintaining a second, drifting schema list here.
//!
//! # Example
//!
//! ```rust,ignore
//! use rmcp_memex::{MemexEngine, MemexConfig};
//! use rmcp_memex::tools::{store_document, search_documents, tool_definitions, ToolResult};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let engine = MemexEngine::for_app("my-app", "documents").await?;
//!
//!     // Store a document using the local helper API
//!     let result = store_document(
//!         &engine,
//!         "doc-1".to_string(),
//!         "Important patient notes about feline diabetes".to_string(),
//!         json!({"patient_id": "P-123", "doc_type": "notes"}),
//!     ).await?;
//!     assert!(result.success);
//!
//!     // Search for documents
//!     let results = search_documents(&engine, "diabetes".to_string(), 5, None).await?;
//!     println!("Found {} documents", results.len());
//!
//!     // Get the canonical MCP tool definitions exposed by rmcp-memex
//!     let tools = tool_definitions();
//!     println!("Available tools: {:?}", tools.iter().map(|t| &t.name).collect::<Vec<_>>());
//!
//!     Ok(())
//! }
//! ```

use crate::engine::{BatchResult, MemexEngine, MetaFilter, StoreItem};
use crate::rag::SearchResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// =============================================================================
// TOOL RESULT
// =============================================================================

/// Result type for tool operations.
///
/// Provides a consistent response format for all tool operations,
/// suitable for MCP tool call responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the operation succeeded
    pub success: bool,
    /// Human-readable message describing the result
    pub message: String,
    /// Optional data payload (operation-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolResult {
    /// Create a success result with just a message
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
        }
    }

    /// Create a success result with data
    pub fn ok_with_data(message: impl Into<String>, data: Value) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: None,
        }
    }
}

// =============================================================================
// TOOL FUNCTIONS
// =============================================================================

/// Store text in memory with automatic embedding generation.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Unique document identifier
/// * `text` - Text content to embed and store
/// * `metadata` - Additional metadata (JSON object)
///
/// # Returns
/// `ToolResult` indicating success or failure
///
/// # Example
///
/// ```rust,ignore
/// let result = store_document(
///     &engine,
///     "visit-123".to_string(),
///     "Patient presented with lethargy...".to_string(),
///     json!({"patient_id": "P-456", "visit_type": "checkup"}),
/// ).await?;
/// ```
pub async fn store_document(
    engine: &MemexEngine,
    id: String,
    text: String,
    metadata: Value,
) -> Result<ToolResult> {
    match engine.store(&id, &text, metadata).await {
        Ok(()) => Ok(ToolResult::ok(format!(
            "Document '{}' stored successfully",
            id
        ))),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to store document '{}': {}",
            id, e
        ))),
    }
}

/// Search memory semantically using vector similarity.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `query` - Search query text
/// * `limit` - Maximum number of results to return
/// * `filter` - Optional metadata filter for narrowing results
///
/// # Returns
/// Vector of `SearchResult` ordered by relevance (highest score first)
///
/// # Example
///
/// ```rust,ignore
/// // Simple search
/// let results = search_documents(&engine, "diabetes symptoms".to_string(), 10, None).await?;
///
/// // Filtered search
/// let filter = MetaFilter::for_patient("P-456");
/// let results = search_documents(&engine, "diabetes".to_string(), 10, Some(filter)).await?;
/// ```
pub async fn search_documents(
    engine: &MemexEngine,
    query: String,
    limit: usize,
    filter: Option<MetaFilter>,
) -> Result<Vec<SearchResult>> {
    match filter {
        Some(f) => engine.search_filtered(&query, f, limit).await,
        None => engine.search(&query, limit).await,
    }
}

/// Get a document by ID.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Document identifier to retrieve
///
/// # Returns
/// `Option<SearchResult>` - The document if found, None otherwise
///
/// # Example
///
/// ```rust,ignore
/// if let Some(doc) = get_document(&engine, "visit-123".to_string()).await? {
///     println!("Found: {}", doc.text);
/// }
/// ```
pub async fn get_document(engine: &MemexEngine, id: String) -> Result<Option<SearchResult>> {
    engine.get(&id).await
}

/// Delete a document by ID.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Document identifier to delete
///
/// # Returns
/// `ToolResult` indicating success or failure, with deletion status
///
/// # Example
///
/// ```rust,ignore
/// let result = delete_document(&engine, "visit-123".to_string()).await?;
/// if result.success {
///     println!("Document deleted");
/// }
/// ```
pub async fn delete_document(engine: &MemexEngine, id: String) -> Result<ToolResult> {
    match engine.delete(&id).await {
        Ok(true) => Ok(ToolResult::ok(format!(
            "Document '{}' deleted successfully",
            id
        ))),
        Ok(false) => Ok(ToolResult::ok_with_data(
            format!("Document '{}' not found", id),
            json!({"deleted": false}),
        )),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to delete document '{}': {}",
            id, e
        ))),
    }
}

/// Batch store multiple documents efficiently.
///
/// More efficient than calling `store_document()` multiple times as embeddings
/// are generated in batches.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `items` - Vector of items to store
///
/// # Returns
/// `BatchResult` with success/failure counts
///
/// # Example
///
/// ```rust,ignore
/// let items = vec![
///     StoreItem::new("doc-1", "First document").with_metadata(json!({"type": "note"})),
///     StoreItem::new("doc-2", "Second document").with_metadata(json!({"type": "note"})),
/// ];
/// let result = store_documents_batch(&engine, items).await?;
/// println!("Stored {} documents", result.success_count);
/// ```
pub async fn store_documents_batch(
    engine: &MemexEngine,
    items: Vec<StoreItem>,
) -> Result<BatchResult> {
    engine.store_batch(items).await
}

/// Delete all documents matching a metadata filter.
///
/// This is the primary method for GDPR-compliant data deletion.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `filter` - Metadata filter specifying which documents to delete
///
/// # Returns
/// `ToolResult` with count of deleted documents
///
/// # Example
///
/// ```rust,ignore
/// // GDPR request - delete all patient data
/// let filter = MetaFilter::for_patient("P-456");
/// let result = delete_documents_by_filter(&engine, filter).await?;
/// if let Some(data) = result.data {
///     println!("Deleted {} documents", data["deleted_count"]);
/// }
/// ```
pub async fn delete_documents_by_filter(
    engine: &MemexEngine,
    filter: MetaFilter,
) -> Result<ToolResult> {
    match engine.delete_by_filter(filter.clone()).await {
        Ok(count) => Ok(ToolResult::ok_with_data(
            format!("Deleted {} documents matching filter", count),
            json!({
                "deleted_count": count,
                "filter": filter,
            }),
        )),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to delete by filter: {}",
            e
        ))),
    }
}

// =============================================================================
// TOOL DEFINITIONS FOR MCP
// =============================================================================

/// MCP tool definition schema.
///
/// This structure mirrors the canonical MCP tool metadata emitted by the shared
/// rmcp-memex transport layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (used for invocation)
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// JSON Schema for the tool's input parameters
    #[serde(rename = "inputSchema", alias = "input_schema")]
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Get the canonical MCP tool definitions exposed by rmcp-memex transports.
///
/// This is derived from the shared transport contract so stdio, HTTP/SSE, and
/// library consumers all see the same tool metadata.
///
/// # Example
///
/// ```rust,ignore
/// let tools = tool_definitions();
/// for tool in &tools {
///     println!("Tool: {} - {}", tool.name, tool.description);
/// }
/// ```
pub fn tool_definitions() -> Vec<ToolDefinition> {
    crate::mcp_protocol::shared_tools_list_result()["tools"]
        .as_array()
        .expect("shared_tools_list_result().tools must be an array")
        .iter()
        .map(|tool| {
            ToolDefinition::new(
                tool["name"]
                    .as_str()
                    .expect("shared MCP tool definition missing name"),
                tool["description"]
                    .as_str()
                    .expect("shared MCP tool definition missing description"),
                tool.get("inputSchema")
                    .cloned()
                    .expect("shared MCP tool definition missing inputSchema"),
            )
        })
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_ok() {
        let result = ToolResult::ok("Success");
        assert!(result.success);
        assert_eq!(result.message, "Success");
        assert!(result.data.is_none());
    }

    #[test]
    fn test_tool_result_ok_with_data() {
        let result = ToolResult::ok_with_data("Success", json!({"count": 42}));
        assert!(result.success);
        assert_eq!(result.message, "Success");
        assert_eq!(result.data.unwrap()["count"], 42);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.success);
        assert_eq!(result.message, "Something went wrong");
        assert!(result.data.is_none());
    }

    #[test]
    fn test_tool_definitions_count() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 14);
    }

    #[test]
    fn test_tool_definitions_names() {
        let tools = tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"health"));
        assert!(names.contains(&"rag_index"));
        assert!(names.contains(&"rag_index_text"));
        assert!(names.contains(&"rag_search"));
        assert!(names.contains(&"memory_upsert"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_get"));
        assert!(names.contains(&"memory_delete"));
        assert!(names.contains(&"memory_purge_namespace"));
        assert!(names.contains(&"namespace_create_token"));
        assert!(names.contains(&"namespace_revoke_token"));
        assert!(names.contains(&"namespace_list_protected"));
        assert!(names.contains(&"namespace_security_status"));
        assert!(names.contains(&"dive"));
    }

    #[test]
    fn test_tool_definitions_have_required_fields() {
        let tools = tool_definitions();

        for tool in tools {
            assert!(!tool.name.is_empty(), "Tool name should not be empty");
            assert!(
                !tool.description.is_empty(),
                "Tool description should not be empty"
            );
            assert!(
                tool.input_schema.is_object(),
                "Input schema should be an object"
            );
            assert!(
                tool.input_schema.get("type").is_some(),
                "Input schema should have a type field"
            );
            assert!(
                tool.input_schema.get("properties").is_some(),
                "Input schema should have properties"
            );
        }
    }

    #[test]
    fn test_tool_definitions_match_shared_mcp_contract() {
        let serialized = serde_json::to_value(tool_definitions()).unwrap();
        assert_eq!(
            serialized,
            crate::mcp_protocol::shared_tools_list_result()["tools"]
        );
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult::ok_with_data("Success", json!({"id": "doc-1"}));
        let json_str = serde_json::to_string(&result).unwrap();

        assert!(json_str.contains("\"success\":true"));
        assert!(json_str.contains("\"message\":\"Success\""));
        assert!(json_str.contains("\"data\""));
    }

    #[test]
    fn test_tool_definition_creation() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            }),
        );

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
        assert!(tool.input_schema["properties"]["input"].is_object());
    }

    #[test]
    fn test_tool_definition_serializes_with_mcp_field_name() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            json!({
                "type": "object",
                "properties": {}
            }),
        );

        let value = serde_json::to_value(tool).unwrap();
        assert!(value.get("inputSchema").is_some());
        assert!(value.get("input_schema").is_none());
    }
}
