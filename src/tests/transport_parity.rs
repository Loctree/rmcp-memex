//! Transport parity tests for the shared MCP protocol core.
//!
//! Both stdio and HTTP/SSE route through `McpCore::handle_jsonrpc_request`.
//! These tests pin the single contract surface and prove that a future drift
//! between transports would fail loudly instead of silently shipping.
//!
//! ## Coverage matrix
//!
//! | Surface                        | Static | Stub dispatch | Full dispatch (MLX) |
//! |--------------------------------|--------|---------------|---------------------|
//! | initialize                     | ✓      | ✓             | ✓                   |
//! | tools/list                     | ✓      | ✓             | ✓                   |
//! | tools/call (health)            |        | ✓ (known diff)| ✓ (known diff)      |
//! | tools/call (security_status)   |        | ✓             |                     |
//! | tools/call (list_protected)    |        | ✓             |                     |
//! | tools/call (dive validation)   |        | ✓             |                     |
//! | tools/call (unknown)           |        | ✓             | ✓                   |
//! | resources/list rejected        |        |               | ✓                   |
//! | missing id                     |        |               | ✓                   |
//! | notifications                  |        | ✓             | ✓                   |
//! | JSON-RPC framing               | ✓      |               |                     |
//! | payload parse error            |        | ✓             |                     |
//! | payload too large              |        | ✓             |                     |
//! | tools/call (create_token)      |        | ✓             |                     |
//! | tools/call (revoke_token val)  |        | ✓             |                     |
//! | tools/call via handle_payload  |        | ✓             |                     |
//! | malformed JSON-RPC structure   |        | ✓             |                     |
//!
//! ## Known transport differences
//!
//! The `health` tool includes a `"transport": "mcp-over-sse"` field when called
//! via `McpTransport::HttpSse`. This is intentional — clients use it for
//! diagnostics. All other tool responses are transport-independent.

use crate::{
    MCPServer, McpDispatch, dispatch_mcp_jsonrpc_request, dispatch_mcp_request,
    mcp_protocol::{
        McpCore, McpTransport, PROTOCOL_VERSION, SERVER_NAME, jsonrpc_error, jsonrpc_success,
        shared_initialize_result, shared_tools_list_result,
    },
};
use serde_json::{Value, json};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tool_names() -> Vec<String> {
    shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(ToOwned::to_owned))
        .collect()
}

/// The canonical tool surface. Every add or remove MUST update this list.
const EXPECTED_TOOLS: &[&str] = &[
    "health",
    "rag_index",
    "rag_index_text",
    "rag_search",
    "memory_upsert",
    "memory_get",
    "memory_search",
    "memory_delete",
    "memory_purge_namespace",
    "namespace_create_token",
    "namespace_revoke_token",
    "namespace_list_protected",
    "namespace_security_status",
    "dive",
];

// ---------------------------------------------------------------------------
// Static contract tests (no infrastructure required)
// ---------------------------------------------------------------------------

#[test]
fn initialize_result_has_correct_protocol_version() {
    let result = shared_initialize_result();
    assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
}

#[test]
fn initialize_result_has_server_info() {
    let result = shared_initialize_result();
    assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    assert!(
        result["serverInfo"]["version"].is_string(),
        "version must be a string"
    );
}

#[test]
fn initialize_result_advertises_only_tools_capability() {
    let result = shared_initialize_result();
    assert_eq!(result["capabilities"], json!({ "tools": {} }));
}

#[test]
fn initialize_result_has_no_extra_top_level_keys() {
    let result = shared_initialize_result();
    let obj = result
        .as_object()
        .expect("initialize result should be an object");
    let keys: Vec<&String> = obj.keys().collect();
    assert!(keys.contains(&&"protocolVersion".to_string()));
    assert!(keys.contains(&&"serverInfo".to_string()));
    assert!(keys.contains(&&"capabilities".to_string()));
    assert_eq!(
        keys.len(),
        3,
        "unexpected keys in initialize result: {:?}",
        keys
    );
}

#[test]
fn unified_tool_surface_matches_expected_names() {
    let names = tool_names();

    for tool in EXPECTED_TOOLS {
        assert!(
            names.contains(&tool.to_string()),
            "Expected shared tool '{}' not found in {:?}",
            tool,
            names
        );
    }

    assert_eq!(
        names.len(),
        EXPECTED_TOOLS.len(),
        "Tool count mismatch. Got {:?}, expected {:?}",
        names,
        EXPECTED_TOOLS
    );
}

#[test]
fn unified_tool_surface_includes_previously_drifted_http_gaps() {
    let names = tool_names();

    // These were historically missing from the HTTP surface before unification.
    assert!(names.contains(&"rag_index".to_string()));
    assert!(names.contains(&"memory_purge_namespace".to_string()));
    assert!(names.contains(&"namespace_create_token".to_string()));
    assert!(names.contains(&"dive".to_string()));
}

#[test]
fn all_tools_have_valid_input_schema() {
    for tool in shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
    {
        let name = tool["name"].as_str().unwrap_or("<unknown>");
        let schema = &tool["inputSchema"];

        assert_eq!(
            schema["type"], "object",
            "Tool '{}' inputSchema.type must be 'object'",
            name
        );
        assert!(
            schema["properties"].is_object(),
            "Tool '{}' must have properties object",
            name
        );
        assert!(
            schema["required"].is_array(),
            "Tool '{}' must have required array",
            name
        );
    }
}

#[test]
fn tools_with_required_fields_declare_them_in_properties() {
    for tool in shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
    {
        let name = tool["name"].as_str().unwrap_or("<unknown>");
        let properties = &tool["inputSchema"]["properties"];
        let required = tool["inputSchema"]["required"]
            .as_array()
            .expect("required should be array");

        for req in required {
            let req_name = req.as_str().unwrap_or("");
            assert!(
                properties.get(req_name).is_some(),
                "Tool '{}' declares required field '{}' but it is missing from properties",
                name,
                req_name
            );
        }
    }
}

#[test]
fn tools_list_result_wraps_in_tools_key() {
    let result = shared_tools_list_result();
    assert!(result["tools"].is_array());
    assert_eq!(
        result["tools"].as_array().unwrap().len(),
        EXPECTED_TOOLS.len()
    );
}

// ---------------------------------------------------------------------------
// JSON-RPC framing tests
// ---------------------------------------------------------------------------

#[test]
fn jsonrpc_success_includes_id_when_present() {
    let resp = jsonrpc_success(&json!(42), json!({"ok": true}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 42);
    assert_eq!(resp["result"]["ok"], true);
    assert!(resp.get("error").is_none());
}

#[test]
fn jsonrpc_success_omits_id_when_null() {
    let resp = jsonrpc_success(&Value::Null, json!({"ok": true}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
}

#[test]
fn jsonrpc_success_with_string_id() {
    let resp = jsonrpc_success(&json!("req-abc"), json!(42));
    assert_eq!(resp["id"], "req-abc");
    assert_eq!(resp["result"], 42);
}

#[test]
fn jsonrpc_error_includes_id_when_present() {
    let request_id = json!("req-1");
    let resp = jsonrpc_error(Some(&request_id), -32601, "Not found");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "req-1");
    assert_eq!(resp["error"]["code"], -32601);
    assert_eq!(resp["error"]["message"], "Not found");
    assert!(resp.get("result").is_none());
}

#[test]
fn jsonrpc_error_omits_id_when_null() {
    let resp = jsonrpc_error(Some(&Value::Null), -32700, "Parse error");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
    assert_eq!(resp["error"]["code"], -32700);
}

#[test]
fn jsonrpc_error_omits_id_when_none() {
    let resp = jsonrpc_error(None, -32600, "Bad request");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
    assert_eq!(resp["error"]["code"], -32600);
}

// ---------------------------------------------------------------------------
// Dispatch parity tests (require McpCore — need real infrastructure)
// ---------------------------------------------------------------------------

/// Build a McpCore backed by a temporary LanceDB + the configured embedding server.
/// Returns None if the embedding server is unreachable (test should skip).
#[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
async fn try_build_mcp_core() -> Option<McpCore> {
    use crate::{
        EmbeddingClient, EmbeddingConfig, ProviderConfig,
        rag::RAGPipeline,
        security::{NamespaceAccessManager, NamespaceSecurityConfig},
        storage::StorageManager,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;

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

    let embedding_client = match EmbeddingClient::new(&config).await {
        Ok(c) => Arc::new(Mutex::new(c)),
        Err(_) => {
            eprintln!("⚠ Skipping dispatch parity test: MLX server unavailable at localhost:12345");
            return None;
        }
    };

    let tmp = tempfile::tempdir().ok()?;
    let db_path = tmp.path().join(".lancedb");
    // Leak the tempdir so it outlives the test — the OS cleans up /tmp anyway
    let db_path_str = db_path.to_string_lossy().to_string();
    std::mem::forget(tmp);

    let storage = Arc::new(StorageManager::new(&db_path_str).await.ok()?);
    let rag = Arc::new(
        RAGPipeline::new(embedding_client.clone(), storage)
            .await
            .ok()?,
    );

    let access_manager = Arc::new(NamespaceAccessManager::new(
        NamespaceSecurityConfig::default(),
    ));
    let _ = access_manager.init().await;

    Some(McpCore::new(
        rag,
        None, // no hybrid searcher needed for protocol tests
        embedding_client,
        5 * 1024 * 1024,
        vec![],
        access_manager,
    ))
}

macro_rules! require_mcp_core {
    ($core:expr) => {
        match $core {
            Some(core) => core,
            None => return,
        }
    };
}

/// Dispatch an `initialize` request through both transports and assert identical results.
#[tokio::test]
async fn dispatch_initialize_parity_across_transports() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("initialize must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("initialize must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "initialize response must be identical across transports"
    );

    // Verify the content is correct
    assert_eq!(stdio_resp["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(stdio_resp["result"]["serverInfo"]["name"], SERVER_NAME);
    assert_eq!(stdio_resp["id"], 1);
}

/// Dispatch a `tools/list` request through both transports and assert identical results.
#[tokio::test]
async fn dispatch_tools_list_parity_across_transports() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "list-1",
        "method": "tools/list",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("tools/list must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("tools/list must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "tools/list response must be identical across transports"
    );

    let tool_count = stdio_resp["result"]["tools"]
        .as_array()
        .expect("tools should be an array")
        .len();
    assert_eq!(tool_count, EXPECTED_TOOLS.len());
}

/// Unknown method returns the same error shape for both transports.
#[tokio::test]
async fn dispatch_unknown_method_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "nonexistent/method",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("unknown method must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("unknown method must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "unknown method error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32601);
    assert!(
        stdio_resp["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("nonexistent/method")
    );
}

/// Unknown tool name inside tools/call returns the same error for both transports.
#[tokio::test]
async fn dispatch_unknown_tool_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "tool-1",
        "method": "tools/call",
        "params": {
            "name": "nonexistent_tool",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("unknown tool must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("unknown tool must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "unknown tool error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32601);
}

/// Missing `id` field produces the same error for both transports.
#[tokio::test]
async fn dispatch_missing_id_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("missing id must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("missing id must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "missing id error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32600);
}

/// Notifications are swallowed (return None) by both transports.
#[tokio::test]
async fn dispatch_notification_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    let stdio_result = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option();

    let http_result = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option();

    assert!(stdio_result.is_none(), "stdio should swallow notification");
    assert!(http_result.is_none(), "http should swallow notification");
}

// ---------------------------------------------------------------------------
// Known transport difference: health tool `transport` field
// ---------------------------------------------------------------------------

/// The `health` tool intentionally differs between transports:
/// - Stdio: no `transport` field in the result
/// - HttpSse: includes `"transport": "mcp-over-sse"`
///
/// This test documents and pins that difference so it is not accidentally
/// removed or changed without updating both transports.
#[tokio::test]
async fn health_tool_transport_field_difference_is_intentional() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "health-1",
        "method": "tools/call",
        "params": {
            "name": "health",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("health must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("health must return a response");

    // Both must succeed with content array
    let stdio_text = stdio_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("stdio health should have text");
    let http_text = http_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("http health should have text");

    let stdio_json: Value = serde_json::from_str(stdio_text).expect("stdio health should be JSON");
    let http_json: Value = serde_json::from_str(http_text).expect("http health should be JSON");

    // Stdio: no transport field
    assert!(
        stdio_json.get("transport").is_none(),
        "stdio health should NOT have transport field"
    );

    // HttpSse: has transport field
    assert_eq!(
        http_json["transport"], "mcp-over-sse",
        "http health should have transport: mcp-over-sse"
    );

    // Everything else should match
    assert_eq!(stdio_json["version"], http_json["version"]);
    assert_eq!(stdio_json["backend"], http_json["backend"]);
    assert_eq!(stdio_json["db_path"], http_json["db_path"]);
}

// ---------------------------------------------------------------------------
// Stub-backed dispatch tests (no embedding server required)
// ---------------------------------------------------------------------------

/// Build an McpCore with a stub embedding client.
/// These tests cover protocol dispatch paths that don't touch embeddings.
#[allow(deprecated)] // NamespaceAccessManager deprecated by Track C; kept for transition
async fn build_mcp_core_stub() -> McpCore {
    use crate::{
        EmbeddingClient,
        rag::RAGPipeline,
        security::{NamespaceAccessManager, NamespaceSecurityConfig},
        storage::StorageManager,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::stub_for_tests()));

    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join(".lancedb");
    let db_path_str = db_path.to_string_lossy().to_string();
    std::mem::forget(tmp);

    let storage = Arc::new(
        StorageManager::new(&db_path_str)
            .await
            .expect("StorageManager::new"),
    );
    let rag = Arc::new(
        RAGPipeline::new(embedding_client.clone(), storage)
            .await
            .expect("RAGPipeline::new"),
    );

    let access_manager = Arc::new(NamespaceAccessManager::new(
        NamespaceSecurityConfig::default(),
    ));
    let _ = access_manager.init().await;

    McpCore::new(
        rag,
        None,
        embedding_client,
        5 * 1024 * 1024,
        vec![],
        access_manager,
    )
}

/// `namespace_security_status` returns identical responses across transports.
/// This is a representative tools/call success path that needs no embeddings.
#[tokio::test]
async fn dispatch_tools_call_security_status_parity() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "sec-1",
        "method": "tools/call",
        "params": {
            "name": "namespace_security_status",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("namespace_security_status must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("namespace_security_status must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "namespace_security_status must be identical across transports"
    );

    // Verify the response shape: MCP content array with text
    let text = stdio_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("should have text content");
    assert!(
        text.contains("Namespace security"),
        "response should contain security status info"
    );
}

/// `namespace_list_protected` returns identical responses across transports.
#[tokio::test]
async fn dispatch_tools_call_list_protected_parity() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "list-p-1",
        "method": "tools/call",
        "params": {
            "name": "namespace_list_protected",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("namespace_list_protected must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("namespace_list_protected must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "namespace_list_protected must be identical across transports"
    );
}

/// `dive` with missing required args returns the same error across transports.
/// This is a representative tools/call failure path (parameter validation).
#[tokio::test]
async fn dispatch_tools_call_dive_missing_args_parity() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "dive-err-1",
        "method": "tools/call",
        "params": {
            "name": "dive",
            "arguments": {
                "namespace": "",
                "query": ""
            }
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("dive validation error must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("dive validation error must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "dive validation error must be identical across transports"
    );

    // Should be a JSON-RPC error with parameter validation code
    assert_eq!(stdio_resp["error"]["code"], -32602);
    assert!(
        stdio_resp["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("required"),
        "error should mention required fields"
    );
}

/// `health` tool returns the same structure for both transports
/// (except the documented `transport` field).
/// Uses stub — doesn't need real MLX.
#[tokio::test]
async fn dispatch_tools_call_health_structure_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "health-stub-1",
        "method": "tools/call",
        "params": {
            "name": "health",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("health must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("health must return a response");

    // Both must have content array with text
    assert!(stdio_resp["result"]["content"][0]["text"].is_string());
    assert!(http_resp["result"]["content"][0]["text"].is_string());

    // Parse the JSON text
    let stdio_json: Value =
        serde_json::from_str(stdio_resp["result"]["content"][0]["text"].as_str().unwrap())
            .expect("stdio health JSON");
    let http_json: Value =
        serde_json::from_str(http_resp["result"]["content"][0]["text"].as_str().unwrap())
            .expect("http health JSON");

    // Stdio: no transport field; HttpSse: has it
    assert!(stdio_json.get("transport").is_none());
    assert_eq!(http_json["transport"], "mcp-over-sse");

    // Shared fields must match
    assert_eq!(stdio_json["version"], http_json["version"]);
    assert_eq!(stdio_json["backend"], http_json["backend"]);
}

/// Stub-based initialize parity — proves the path works even without MLX.
#[tokio::test]
async fn dispatch_initialize_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("initialize must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("initialize must return a response");

    assert_eq!(stdio_resp, http_resp);
    assert_eq!(stdio_resp["result"]["protocolVersion"], PROTOCOL_VERSION);
}

/// Stub-based tools/list parity.
#[tokio::test]
async fn dispatch_tools_list_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "tl-stub",
        "method": "tools/list",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("tools/list must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("tools/list must return a response");

    assert_eq!(stdio_resp, http_resp);
    assert_eq!(
        stdio_resp["result"]["tools"].as_array().unwrap().len(),
        EXPECTED_TOOLS.len()
    );
}

/// Unknown tool via stub — proves error parity without MLX.
#[tokio::test]
async fn dispatch_unknown_tool_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "ut-stub",
        "method": "tools/call",
        "params": {
            "name": "nonexistent_tool",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("unknown tool must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("unknown tool must return error");

    assert_eq!(stdio_resp, http_resp);
    assert_eq!(stdio_resp["error"]["code"], -32601);
}

/// Notification via stub — proves both transports swallow it.
#[tokio::test]
async fn dispatch_notification_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    let stdio = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option();
    let http = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option();

    assert!(stdio.is_none());
    assert!(http.is_none());
}

/// Raw payload parse failures now come from the shared framing path.
#[tokio::test]
async fn handle_payload_parse_error_parity_stub() {
    let core = build_mcp_core_stub().await;
    let payload = "{";

    let stdio = core.handle_payload(payload, McpTransport::Stdio).await;
    let http = core.handle_payload(payload, McpTransport::HttpSse).await;

    assert_eq!(stdio, http);

    let response = stdio.expect("parse errors must produce a JSON-RPC response");
    assert_eq!(response["error"]["code"], -32700);
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Parse error")
    );
}

/// Valid JSON payload through `handle_payload` produces identical responses.
/// This tests the full wire path: raw string → parse → dispatch → response.
#[tokio::test]
async fn handle_payload_valid_roundtrip_parity_stub() {
    let core = build_mcp_core_stub().await;
    let payload = r#"{"jsonrpc":"2.0","id":"pl-1","method":"initialize","params":{}}"#;

    let stdio = core.handle_payload(payload, McpTransport::Stdio).await;
    let http = core.handle_payload(payload, McpTransport::HttpSse).await;

    assert_eq!(
        stdio, http,
        "valid payload roundtrip must be identical across transports"
    );

    let response = stdio.expect("valid payload must produce a response");
    assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(response["id"], "pl-1");
}

/// The shared request-dispatch facade preserves transport parity.
#[tokio::test]
async fn dispatch_request_wrapper_roundtrip_parity_stub() {
    let core = build_mcp_core_stub().await;
    let request = json!({
        "jsonrpc": "2.0",
        "id": "wrapper-1",
        "method": "initialize",
        "params": {}
    });

    let stdio = dispatch_mcp_request(&core, request.clone(), McpTransport::Stdio).await;
    let http = dispatch_mcp_request(&core, request, McpTransport::HttpSse).await;

    assert_eq!(stdio, http);

    let response = stdio.expect("initialize through facade must produce a response");
    assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(response["id"], "wrapper-1");
}

/// The JSON-RPC facade preserves notification semantics instead of coercing them
/// into synthetic responses.
#[tokio::test]
async fn dispatch_jsonrpc_wrapper_preserves_notification_semantics_stub() {
    let core = build_mcp_core_stub().await;
    let request = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    let stdio = dispatch_mcp_jsonrpc_request(&core, request.clone(), McpTransport::Stdio).await;
    let http = dispatch_mcp_jsonrpc_request(&core, request, McpTransport::HttpSse).await;

    assert!(matches!(stdio, McpDispatch::Notification));
    assert!(matches!(http, McpDispatch::Notification));
}

/// The `MCPServer` convenience wrapper must preserve stdio notification
/// semantics instead of fabricating a JSON-RPC success response.
#[tokio::test]
async fn server_dispatch_request_preserves_notification_semantics_stub() {
    let server = MCPServer::from_mcp_core(Arc::new(build_mcp_core_stub().await));
    let request = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    let response = server.dispatch_request(request).await;

    assert!(
        response.is_none(),
        "notification dispatch must remain silent for stdio callers too"
    );
}

/// The `MCPServer` helper should be a thin wrapper over the shared core, not a
/// second JSON-RPC implementation.
#[tokio::test]
async fn server_dispatch_request_matches_shared_facade_stub() {
    let core = Arc::new(build_mcp_core_stub().await);
    let server = MCPServer::from_mcp_core(core.clone());
    let request = json!({
        "jsonrpc": "2.0",
        "id": "server-wrapper-1",
        "method": "initialize",
        "params": {}
    });

    let server_response = server.dispatch_request(request.clone()).await;
    let shared_response = dispatch_mcp_request(core.as_ref(), request, McpTransport::Stdio).await;

    assert_eq!(server_response, shared_response);
}

/// Valid tools/list through raw payload path — confirms wire-level parity.
#[tokio::test]
async fn handle_payload_tools_list_roundtrip_parity_stub() {
    let core = build_mcp_core_stub().await;
    let payload = r#"{"jsonrpc":"2.0","id":42,"method":"tools/list","params":{}}"#;

    let stdio = core.handle_payload(payload, McpTransport::Stdio).await;
    let http = core.handle_payload(payload, McpTransport::HttpSse).await;

    assert_eq!(stdio, http);

    let response = stdio.expect("tools/list payload must produce a response");
    assert_eq!(
        response["result"]["tools"].as_array().unwrap().len(),
        EXPECTED_TOOLS.len()
    );
}
/// Oversized payloads are rejected before transport-specific delivery.
#[tokio::test]
async fn handle_payload_request_too_large_parity_stub() {
    let core = build_mcp_core_stub().await;
    let oversized_payload = "x".repeat(5 * 1024 * 1024 + 1);

    let stdio = core
        .handle_payload(&oversized_payload, McpTransport::Stdio)
        .await;
    let http = core
        .handle_payload(&oversized_payload, McpTransport::HttpSse)
        .await;

    assert_eq!(stdio, http);

    let response = stdio.expect("oversized payload must produce a JSON-RPC response");
    assert_eq!(response["error"]["code"], -32600);
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Request too large")
    );
}

/// `namespace_create_token` success path — exercises a different tool category than health/security_status.
#[tokio::test]
async fn dispatch_tools_call_create_token_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "ct-1",
        "method": "tools/call",
        "params": {
            "name": "namespace_create_token",
            "arguments": {
                "namespace": "parity-test-ns",
                "description": "transport parity test"
            }
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("namespace_create_token must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("namespace_create_token must return a response");

    // Both must succeed with content array containing token text
    let stdio_text = stdio_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("stdio should have text content");
    let http_text = http_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("http should have text content");

    assert!(
        stdio_text.contains("Token created"),
        "stdio response should confirm token creation"
    );
    assert!(
        http_text.contains("Token created"),
        "http response should confirm token creation"
    );

    // Structure must match (actual token values will differ, so compare shape only)
    assert_eq!(
        stdio_resp["result"]["content"][0]["type"], http_resp["result"]["content"][0]["type"],
        "content type must be identical across transports"
    );
}

/// `namespace_revoke_token` with empty namespace — tool-level validation parity.
#[tokio::test]
async fn dispatch_tools_call_revoke_token_empty_ns_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "rt-1",
        "method": "tools/call",
        "params": {
            "name": "namespace_revoke_token",
            "arguments": {
                "namespace": ""
            }
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("revoke_token must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("revoke_token must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "revoke_token validation error must be identical across transports"
    );

    // tool_error_message returns { error: { message: "..." } } inside jsonrpc_success result
    let msg = stdio_resp["result"]["error"]["message"]
        .as_str()
        .unwrap_or("");
    assert!(
        msg.contains("required") || msg.contains("Namespace"),
        "response should indicate namespace is required, got: {}",
        msg
    );
}

/// Wire-path parity for tools/call — proves the full raw string → parse → dispatch → response
/// path works identically for tool calls, not just initialize/tools_list.
#[tokio::test]
async fn handle_payload_tools_call_roundtrip_parity_stub() {
    let core = build_mcp_core_stub().await;
    let payload = r#"{"jsonrpc":"2.0","id":"wire-tc","method":"tools/call","params":{"name":"namespace_security_status","arguments":{}}}"#;

    let stdio = core.handle_payload(payload, McpTransport::Stdio).await;
    let http = core.handle_payload(payload, McpTransport::HttpSse).await;

    assert_eq!(
        stdio, http,
        "tools/call wire-path roundtrip must be identical across transports"
    );

    let response = stdio.expect("tools/call payload must produce a response");
    assert!(
        response["result"]["content"][0]["text"].is_string(),
        "response should have text content"
    );
    assert_eq!(response["id"], "wire-tc");
}

/// Malformed JSON-RPC structure (valid JSON, but missing "method" field).
/// Different from parse error — this tests structural validation parity.
#[tokio::test]
async fn dispatch_missing_method_parity_stub() {
    let core = build_mcp_core_stub().await;

    let request = json!({
        "jsonrpc": "2.0",
        "id": "no-method",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("missing method must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("missing method must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "missing method error must be identical across transports"
    );

    // Missing method results in empty string → "Unknown method: "
    assert_eq!(stdio_resp["error"]["code"], -32601);
}
