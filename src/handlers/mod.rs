use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::{
    ServerConfig,
    mcp_core::{
        McpCore, McpDispatch, McpTransport, dispatch_mcp_jsonrpc_request, dispatch_mcp_payload,
    },
    mcp_runtime::build_mcp_core,
};

pub struct MCPServer {
    mcp_core: Arc<McpCore>,
}

impl MCPServer {
    /// Get the shared MCP core for reuse across transports.
    pub fn mcp_core(&self) -> Arc<McpCore> {
        self.mcp_core.clone()
    }

    pub async fn run_stdio(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let read = reader.read_line(&mut line).await?;
            if read == 0 {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(response) =
                dispatch_mcp_payload(self.mcp_core.as_ref(), trimmed, McpTransport::Stdio).await
            {
                write_json_line(&mut stdout, &response).await?;
            }
        }

        Ok(())
    }

    pub async fn run(self) -> Result<()> {
        self.run_stdio().await
    }

    /// Dispatch a parsed JSON-RPC request through the shared stdio MCP core.
    ///
    /// Notifications return `None` so library callers see the same semantics as
    /// the real stdio transport instead of a synthetic success response.
    pub async fn dispatch_request(&self, request: Value) -> Option<Value> {
        self.dispatch_jsonrpc_request(request).await.into_option()
    }

    /// Dispatch a parsed JSON-RPC request and preserve notification semantics.
    pub async fn dispatch_jsonrpc_request(&self, request: Value) -> McpDispatch {
        dispatch_mcp_jsonrpc_request(self.mcp_core.as_ref(), request, McpTransport::Stdio).await
    }

    #[cfg(test)]
    pub(crate) fn from_mcp_core(mcp_core: Arc<McpCore>) -> Self {
        Self { mcp_core }
    }
}

pub async fn create_server(config: ServerConfig) -> Result<MCPServer> {
    let mcp_core = build_mcp_core(config).await?;

    Ok(MCPServer { mcp_core })
}

async fn write_json_line(stdout: &mut tokio::io::Stdout, response: &Value) -> anyhow::Result<()> {
    let payload = serde_json::to_string(response)?;
    stdout.write_all(payload.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}
