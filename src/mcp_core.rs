//! Shared MCP transport entry points.
//!
//! `mcp_protocol` owns the protocol contract and tool dispatch implementation.
//! This module is the transport-facing facade that both stdio and HTTP/SSE call
//! so request parsing and dispatch stay routed through one public surface.

use serde_json::Value;

pub use crate::mcp_protocol::{
    McpCore, McpDispatch, McpTransport, shared_initialize_result, shared_tools_list_result,
};

/// Dispatch a parsed JSON-RPC request through the shared MCP core.
pub async fn dispatch_mcp_request(
    mcp_core: &McpCore,
    request: Value,
    transport: McpTransport,
) -> Option<Value> {
    mcp_core.handle_request(request, transport).await
}

/// Dispatch a raw JSON-RPC payload through the shared MCP core.
pub async fn dispatch_mcp_payload(
    mcp_core: &McpCore,
    payload: &str,
    transport: McpTransport,
) -> Option<Value> {
    mcp_core.handle_payload(payload, transport).await
}

/// Dispatch a parsed JSON-RPC request and preserve notification semantics.
pub async fn dispatch_mcp_jsonrpc_request(
    mcp_core: &McpCore,
    request: Value,
    transport: McpTransport,
) -> McpDispatch {
    mcp_core.handle_jsonrpc_request(request, transport).await
}
