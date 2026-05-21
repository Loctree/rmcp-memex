// rmcp-memex
//
// The canonical custom Rust MCP kernel for RAG and long-term memory.
// This binary exposes the primary surface of the rmcp-memex engine.
//
// It supports two explicit transport modes:
// - stdio: Native MCP integration (e.g., for Claude Desktop)
// - HTTP/SSE: Multi-agent daemon mode
//
// `rmcp-memex` is the only supported binary name. The GitHub installer may
// also create `rmcp_memex` as a legacy compatibility symlink for older scripts.
//
// Usage:
//   rmcp-memex <command> [args]
//
// Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

use anyhow::Result;
use clap::Parser;

pub mod cli;
use crate::cli::definition::Cli;
use crate::cli::dispatch::run_command;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    run_command(cli).await
}
