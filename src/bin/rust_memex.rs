//! rust-memex CLI - Vector memory management tool
//!
//! This binary is an alias for rmcp-memex, providing a CLI-focused name.
//! Both binaries share the same codebase.
//!
//! Usage:
//!   rust-memex index <path>                - Index documents
//!   rust-memex search -n <ns> -q <query>   - Search documents
//!   rust-memex audit                       - Check database quality
//!   rust-memex purge-quality               - Remove low-quality namespaces
//!
//! For MCP server: rust-memex serve (or use rmcp-memex)
//!
//! Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

// Include the same main code as rmcp_memex.rs
// This allows both binaries to share the implementation
include!("rmcp_memex.rs");
