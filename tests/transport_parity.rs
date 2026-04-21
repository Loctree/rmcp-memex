//! Transport Parity Integration Tests
//!
//! Black-box tests proving that stdio and SSE/HTTP transports expose the
//! same MCP contract. Every test dispatches identical JSON-RPC payloads
//! through both `McpTransport::Stdio` and `McpTransport::HttpSse` and
//! asserts structurally equivalent responses.
//!
//! These tests require a running embedding server (Ollama with qwen3-embedding:8b, 4096 dims).
//! Run with: cargo test --test transport_parity -- --ignored
//!
//! Vibecrafted with AI Agents (c)2026 VetCoders

use rmcp_memex::{
    EmbeddingConfig, HybridConfig, McpTransport, NamespaceSecurityConfig, ProviderConfig,
    ServerConfig, build_mcp_core, dispatch_mcp_payload,
};
use serde_json::{Value, json};
use tempfile::TempDir;

const LOCAL_OLLAMA_MODEL: &str = "qwen3-embedding:8b";
const LOCAL_OLLAMA_DIMENSION: usize = 4096;

// =============================================================================
// Helpers
// =============================================================================

async fn ollama_available() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    let Ok(resp) = client.get("http://localhost:11434/api/tags").send().await else {
        return false;
    };
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return false;
    };
    // Check the specific model is pulled, not just that Ollama is running
    body["models"].as_array().is_some_and(|models| {
        models.iter().any(|m| {
            m["name"]
                .as_str()
                .is_some_and(|n| n.starts_with(LOCAL_OLLAMA_MODEL))
        })
    })
}

fn test_config(db_path: &str) -> ServerConfig {
    ServerConfig {
        cache_mb: 64,
        db_path: db_path.to_string(),
        max_request_bytes: 1024 * 1024,
        log_level: tracing::Level::WARN,
        allowed_paths: vec![],
        security: NamespaceSecurityConfig::default(),
        embeddings: EmbeddingConfig {
            required_dimension: LOCAL_OLLAMA_DIMENSION,
            max_batch_chars: 32000,
            max_batch_items: 16,
            providers: vec![ProviderConfig {
                name: "ollama-local".into(),
                base_url: "http://localhost:11434".into(),
                model: LOCAL_OLLAMA_MODEL.into(),
                priority: 1,
                endpoint: "/v1/embeddings".into(),
            }],
            reranker: Default::default(),
        },
        hybrid: HybridConfig::default(),
    }
}

/// Dispatch same payload through both transports, return (stdio_resp, sse_resp).
async fn dispatch_both(
    core: &rmcp_memex::McpCore,
    payload: &str,
) -> (Option<Value>, Option<Value>) {
    let stdio = dispatch_mcp_payload(core, payload, McpTransport::Stdio).await;
    let sse = dispatch_mcp_payload(core, payload, McpTransport::HttpSse).await;
    (stdio, sse)
}

/// Strip the known transport-dependent `transport` field from health results
/// so we can compare the rest.
fn strip_transport_field(mut value: Value) -> Value {
    if let Some(content) = value["result"]["content"].as_array()
        && let Some(first) = content.first()
        && let Some(text) = first["text"].as_str()
        && let Ok(mut parsed) = serde_json::from_str::<Value>(text)
    {
        parsed.as_object_mut().map(|o| o.remove("transport"));
        let stripped_text = serde_json::to_string(&parsed).unwrap();
        value["result"]["content"][0]["text"] = json!(stripped_text);
    }
    value
}

// =============================================================================
// Parity Tests
// =============================================================================

#[tokio::test]
#[ignore]
async fn parity_initialize_identical_across_transports() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to initialize");
    let sse = sse.expect("sse must respond to initialize");

    assert_eq!(
        stdio, sse,
        "initialize responses must be identical across transports"
    );
    assert_eq!(stdio["jsonrpc"], "2.0");
    assert_eq!(stdio["id"], 1);
    assert!(stdio["result"]["protocolVersion"].is_string());
    assert_eq!(stdio["result"]["capabilities"], json!({ "tools": {} }));
}

#[tokio::test]
#[ignore]
async fn parity_tools_list_identical_across_transports() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to tools/list");
    let sse = sse.expect("sse must respond to tools/list");

    assert_eq!(
        stdio, sse,
        "tools/list responses must be identical across transports"
    );

    let tools = stdio["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 14, "must expose all 14 tools");
}

#[tokio::test]
#[ignore]
async fn parity_health_tool_structurally_equivalent() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"health","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to health");
    let sse = sse.expect("sse must respond to health");

    // Responses differ ONLY in the transport field. Strip it and compare.
    let stdio_stripped = strip_transport_field(stdio.clone());
    let sse_stripped = strip_transport_field(sse.clone());
    assert_eq!(
        stdio_stripped, sse_stripped,
        "health responses must match after stripping transport field"
    );

    // Verify the documented transport-dependent field
    let stdio_text: Value =
        serde_json::from_str(stdio["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let sse_text: Value =
        serde_json::from_str(sse["result"]["content"][0]["text"].as_str().unwrap()).unwrap();

    assert!(
        stdio_text.get("transport").is_none(),
        "stdio health must NOT include transport field"
    );
    assert_eq!(
        sse_text["transport"], "mcp-over-sse",
        "SSE health must include transport: mcp-over-sse"
    );
}

#[tokio::test]
#[ignore]
async fn parity_namespace_security_status_identical() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"namespace_security_status","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond");
    let sse = sse.expect("sse must respond");
    assert_eq!(
        stdio, sse,
        "namespace_security_status must be identical across transports"
    );
}

#[tokio::test]
#[ignore]
async fn parity_resources_list_rejected_until_resources_exist() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":5,"method":"resources/list","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(
        stdio, sse,
        "resources/list must fail identically across transports until resources are implemented"
    );
    assert_eq!(stdio["error"]["code"], -32601);
    assert_eq!(stdio["error"]["message"], "Unknown method: resources/list");
}

#[tokio::test]
#[ignore]
async fn parity_unknown_tool_error_identical() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nonexistent_tool","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(stdio, sse, "unknown tool errors must be identical");
    assert_eq!(stdio["error"]["code"], -32601);
}

#[tokio::test]
#[ignore]
async fn parity_invalid_json_error_identical() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{not valid json"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with parse error");
    let sse = sse.expect("sse must respond with parse error");
    assert_eq!(
        stdio, sse,
        "parse errors must be identical across transports"
    );
    assert_eq!(stdio["error"]["code"], -32700);
}

#[tokio::test]
#[ignore]
async fn parity_missing_id_error_identical() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","method":"initialize","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(stdio, sse, "missing id errors must be identical");
    assert_eq!(stdio["error"]["code"], -32600);
}

#[tokio::test]
#[ignore]
async fn parity_notification_silent_on_both_transports() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama unavailable");
        return;
    }

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    // Notifications must return None on both transports (no response per JSON-RPC spec)
    let payload = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    assert!(stdio.is_none(), "stdio must not respond to notifications");
    assert!(sse.is_none(), "sse must not respond to notifications");
}
