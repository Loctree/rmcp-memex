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
//! - GET  /                  - HTML Dashboard (browse memories visually)
//! - GET  /api/discovery     - Endpoint discovery: status, db info, namespaces (canonical)
//! - GET  /api/namespaces    - List all namespaces with counts
//! - GET  /api/overview      - Database overview/stats
//! - GET  /api/browse/:ns    - Browse documents in namespace
//! - GET  /health            - Health check
//! - POST /search            - Search documents
//! - GET  /sse/search        - SSE streaming search
//! - GET  /sse/namespaces    - SSE streaming namespace listing with summaries
//! - POST /sse/optimize      - SSE streaming database optimize (compact + prune)
//! - POST /upsert            - Upsert document (memory_upsert)
//! - POST /index             - Index text with full pipeline
//! - GET  /expand/:ns/:id    - Expand onion slice (get children)
//! - GET  /parent/:ns/:id    - Get parent slice (drill up)
//! - DELETE /ns/:namespace   - Purge namespace
//!
//! MCP-over-SSE endpoints (for Claude Code compatibility):
//! - GET  /mcp/             - SSE stream for MCP messages (sends endpoint event)
//! - POST /mcp/messages/    - JSON-RPC POST endpoint with session_id
//!
//! Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, Request, State},
    http::{HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{
        Html, IntoResponse,
        sse::{Event, Sse},
    },
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{RwLock, broadcast};
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};

use crate::rag::{RAGPipeline, SearchResult, SliceLayer};

// ============================================================================
// HTML Dashboard (embedded)
// ============================================================================

/// Embedded HTML dashboard for browsing memories visually
const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rmcp-memex Dashboard</title>
    <style>
        :root {
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --accent-muted: #388bfd;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }
        h1 { font-size: 24px; font-weight: 600; }
        h1 span { color: var(--accent); }
        .stats-bar {
            display: flex;
            gap: 24px;
            font-size: 14px;
            color: var(--text-muted);
        }
        .stats-bar strong { color: var(--text); }

        /* Search box */
        .search-box {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .search-box input {
            flex: 1;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 16px;
        }
        .search-box input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .search-box select {
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 14px;
            min-width: 200px;
        }
        .search-box button {
            padding: 12px 24px;
            background: var(--accent);
            border: none;
            border-radius: 6px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .search-box button:hover { background: var(--accent-muted); }

        /* Layout */
        .layout {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 24px;
        }

        /* Sidebar */
        .sidebar {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: fit-content;
            position: sticky;
            top: 20px;
        }
        .sidebar h3 {
            font-size: 14px;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .namespace-list { list-style: none; }
        .namespace-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .namespace-item:hover { background: var(--bg); }
        .namespace-item.active { background: var(--accent); color: #fff; }
        .namespace-item .name { font-weight: 500; font-size: 14px; }
        .namespace-item .count {
            background: var(--bg);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            color: var(--text-muted);
        }
        .namespace-item.active .count { background: rgba(255,255,255,0.2); color: #fff; }

        /* Main content */
        .main { min-width: 0; }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .results-header h2 { font-size: 18px; }
        .results-count { color: var(--text-muted); font-size: 14px; }

        /* Document cards */
        .doc-list { display: flex; flex-direction: column; gap: 12px; }
        .doc-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            transition: border-color 0.2s;
        }
        .doc-card:hover { border-color: var(--accent); }
        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        .doc-id {
            font-family: monospace;
            font-size: 12px;
            color: var(--accent);
            background: var(--bg);
            padding: 4px 8px;
            border-radius: 4px;
        }
        .doc-score {
            font-size: 12px;
            color: var(--success);
            font-weight: 600;
        }
        .doc-text {
            font-size: 14px;
            line-height: 1.6;
            color: var(--text);
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        .doc-meta {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            font-size: 12px;
            color: var(--text-muted);
        }
        .doc-meta .layer {
            padding: 2px 8px;
            background: var(--bg);
            border-radius: 4px;
        }
        .doc-actions {
            margin-top: 12px;
            display: flex;
            gap: 8px;
        }
        .doc-actions button {
            padding: 6px 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-muted);
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .doc-actions button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* Loading state */
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }
        .empty-state h3 { margin-bottom: 8px; color: var(--text); }

        /* Detail modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow: auto;
            padding: 24px;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .modal-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 24px;
            cursor: pointer;
        }
        .modal-close:hover { color: var(--text); }
        .modal pre {
            background: var(--bg);
            padding: 16px;
            border-radius: 8px;
            overflow: auto;
            font-size: 13px;
            white-space: pre-wrap;
        }

        /* Timeline view */
        .timeline { padding: 20px 0; }
        .timeline-item {
            display: flex;
            gap: 16px;
            padding: 12px 0;
            border-left: 2px solid var(--border);
            padding-left: 20px;
            margin-left: 8px;
            position: relative;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -6px;
            top: 18px;
            width: 10px;
            height: 10px;
            background: var(--accent);
            border-radius: 50%;
        }
        .timeline-date {
            min-width: 100px;
            font-size: 12px;
            color: var(--text-muted);
        }

        /* Footer */
        footer {
            margin-top: 40px;
            padding: 20px 0;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>rmcp-<span>memex</span></h1>
            <div class="stats-bar" id="stats-bar">
                <span>Loading...</span>
            </div>
        </header>

        <div class="search-box">
            <input type="text" id="search-input" placeholder="Search memories..." autocomplete="off">
            <select id="namespace-select">
                <option value="">All namespaces</option>
            </select>
            <button onclick="doSearch()">Search</button>
        </div>

        <div class="layout">
            <aside class="sidebar">
                <h3>Namespaces</h3>
                <ul class="namespace-list" id="namespace-list">
                    <li class="loading">Loading...</li>
                </ul>
            </aside>

            <main class="main">
                <div class="results-header">
                    <h2 id="results-title">Recent Memories</h2>
                    <span class="results-count" id="results-count"></span>
                </div>
                <div class="doc-list" id="doc-list">
                    <div class="loading">Loading memories...</div>
                </div>
            </main>
        </div>

        <footer>
            rmcp-memex v{VERSION} | Vibecrafted with AI Agents by VetCoders &copy;2026 VetCoders
        </footer>
    </div>

    <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3 id="modal-title">Document Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <pre id="modal-content"></pre>
        </div>
    </div>

    <script>
        const API = window.location.origin;
        let currentNamespace = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            await loadOverview();
            await loadNamespaces();
            await browse(null);

            // Enter key to search
            document.getElementById('search-input').addEventListener('keypress', e => {
                if (e.key === 'Enter') doSearch();
            });
        });

        // Fetch with timeout helper
        async function fetchWithTimeout(url, options = {}, timeout = 60000) {
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), timeout);
            try {
                const response = await fetch(url, { ...options, signal: controller.signal });
                clearTimeout(id);
                return response;
            } catch (e) {
                clearTimeout(id);
                throw e;
            }
        }

        async function loadOverview() {
            try {
                document.getElementById('stats-bar').innerHTML = '<span>Loading stats...</span>';
                const res = await fetchWithTimeout(`${API}/api/overview`, {}, 120000);
                const data = await res.json();
                document.getElementById('stats-bar').innerHTML = `
                    <span>Namespaces: <strong>${data.namespace_count || '?'}</strong></span>
                    <span>Documents: <strong>${data.total_documents.toLocaleString()}</strong></span>
                    <span>DB: <strong>${data.db_path}</strong></span>
                `;
            } catch (e) {
                document.getElementById('stats-bar').innerHTML = '<span style="color:var(--warning)">Stats slow - run "make optimize"</span>';
            }
        }

        async function loadNamespaces() {
            try {
                // First check cache status
                const statusRes = await fetchWithTimeout(`${API}/api/status`, {}, 5000);
                const status = await statusRes.json();

                const list = document.getElementById('namespace-list');
                const select = document.getElementById('namespace-select');

                if (!status.cache_ready) {
                    // Cache not ready - show loading with hint
                    list.innerHTML = `
                        <li class="empty-state" style="text-align:left;padding:16px;">
                            <h3 style="color:var(--warning)">⏳ Loading namespaces...</h3>
                            <p style="margin-top:8px;font-size:13px;color:var(--text-muted)">
                                Background task is scanning the database.<br>
                                If this persists, run: <code style="color:var(--accent)">rmcp-memex optimize</code>
                            </p>
                        </li>`;
                    // Auto-retry in 5 seconds
                    setTimeout(() => loadNamespaces(), 5000);
                    return;
                }

                const res = await fetchWithTimeout(`${API}/api/namespaces`, {}, 30000);
                const data = await res.json();

                if (data.namespaces.length === 0) {
                    list.innerHTML = '<li class="empty-state"><h3>No namespaces</h3></li>';
                    return;
                }

                list.innerHTML = data.namespaces.map(ns => `
                    <li class="namespace-item${currentNamespace === ns.name ? ' active' : ''}"
                        onclick="selectNamespace('${ns.name}')">
                        <span class="name">${ns.name}</span>
                        <span class="count">${ns.count.toLocaleString()}</span>
                    </li>
                `).join('');

                select.innerHTML = '<option value="">All namespaces</option>' +
                    data.namespaces.map(ns => `<option value="${ns.name}">${ns.name} (${ns.count})</option>`).join('');

            } catch (e) {
                document.getElementById('namespace-list').innerHTML =
                    '<li style="color:var(--error)">Failed to load namespaces</li>';
            }
        }

        async function selectNamespace(ns) {
            currentNamespace = ns;
            document.getElementById('namespace-select').value = ns || '';
            await loadNamespaces();
            await browse(ns);
        }

        async function browse(namespace) {
            const list = document.getElementById('doc-list');
            list.innerHTML = '<div class="loading">Loading documents (large DB may be slow)...</div>';

            try {
                const ns = namespace || '';
                const res = await fetchWithTimeout(`${API}/api/browse/${ns}?limit=50`, {}, 120000);
                const data = await res.json();

                document.getElementById('results-title').textContent =
                    namespace ? `Browsing: ${namespace}` : 'All Memories';
                document.getElementById('results-count').textContent =
                    `${data.documents.length} documents`;

                if (data.documents.length === 0) {
                    list.innerHTML = `
                        <div class="empty-state">
                            <h3>No documents found</h3>
                            <p>This namespace is empty or no data has been indexed yet.</p>
                        </div>
                    `;
                    return;
                }

                list.innerHTML = data.documents.map(doc => renderDocCard(doc)).join('');
            } catch (e) {
                list.innerHTML = `<div class="empty-state" style="color:var(--error)">
                    <h3>Error loading documents</h3>
                    <p>${e.message}</p>
                </div>`;
            }
        }

        async function doSearch() {
            const query = document.getElementById('search-input').value.trim();
            if (!query) {
                await browse(currentNamespace);
                return;
            }

            const list = document.getElementById('doc-list');
            list.innerHTML = '<div class="loading">Searching...</div>';

            const namespace = document.getElementById('namespace-select').value || null;

            try {
                const res = await fetch(`${API}/search`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, namespace, limit: 20 })
                });
                const data = await res.json();

                document.getElementById('results-title').textContent = `Search: "${query}"`;
                document.getElementById('results-count').textContent =
                    `${data.count} results in ${data.elapsed_ms}ms`;

                if (data.results.length === 0) {
                    list.innerHTML = `
                        <div class="empty-state">
                            <h3>No results found</h3>
                            <p>Try a different query or search all namespaces.</p>
                        </div>
                    `;
                    return;
                }

                list.innerHTML = data.results.map(doc => renderDocCard(doc, true)).join('');
            } catch (e) {
                list.innerHTML = `<div class="empty-state" style="color:var(--error)">
                    <h3>Search failed</h3>
                    <p>${e.message}</p>
                </div>`;
            }
        }

        function renderDocCard(doc, showScore = false) {
            const text = doc.text || '';
            const truncated = text.length > 500 ? text.slice(0, 500) + '...' : text;
            const layer = doc.layer || 'flat';

            return `
                <div class="doc-card">
                    <div class="doc-header">
                        <span class="doc-id">${doc.id}</span>
                        ${showScore ? `<span class="doc-score">Score: ${doc.score.toFixed(3)}</span>` : ''}
                    </div>
                    <div class="doc-text">${escapeHtml(truncated)}</div>
                    <div class="doc-meta">
                        <span>Namespace: <strong>${doc.namespace}</strong></span>
                        <span class="layer">${layer}</span>
                        ${doc.can_expand ? '<span style="color:var(--accent)">▼ Has children</span>' : ''}
                        ${doc.can_drill_up ? '<span style="color:var(--warning)">▲ Has parent</span>' : ''}
                    </div>
                    <div class="doc-actions">
                        <button onclick='showDetails(${JSON.stringify(doc).replace(/'/g, "&#39;")})'>Details</button>
                        ${doc.can_expand ? `<button onclick="expand('${doc.namespace}', '${doc.id}')">Expand ▼</button>` : ''}
                        ${doc.can_drill_up ? `<button onclick="drillUp('${doc.namespace}', '${doc.id}')">Parent ▲</button>` : ''}
                    </div>
                </div>
            `;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function expand(ns, id) {
            const list = document.getElementById('doc-list');
            const oldContent = list.innerHTML;
            list.innerHTML = '<div class="loading">Expanding...</div>';

            try {
                const res = await fetch(`${API}/expand/${ns}/${id}`);
                const data = await res.json();

                document.getElementById('results-title').textContent = `Children of: ${id}`;
                document.getElementById('results-count').textContent = `${data.count} children`;

                if (data.children.length === 0) {
                    list.innerHTML = `<div class="empty-state"><h3>No children</h3></div>`;
                    return;
                }

                list.innerHTML = data.children.map(doc => renderDocCard(doc)).join('');
            } catch (e) {
                list.innerHTML = oldContent;
                alert('Failed to expand: ' + e.message);
            }
        }

        async function drillUp(ns, id) {
            const list = document.getElementById('doc-list');
            const oldContent = list.innerHTML;
            list.innerHTML = '<div class="loading">Finding parent...</div>';

            try {
                const res = await fetch(`${API}/parent/${ns}/${id}`);
                const data = await res.json();

                document.getElementById('results-title').textContent = `Parent of: ${id}`;
                document.getElementById('results-count').textContent = '1 document';

                list.innerHTML = renderDocCard(data.parent);
            } catch (e) {
                list.innerHTML = oldContent;
                alert('Failed to find parent: ' + e.message);
            }
        }

        function showDetails(doc) {
            document.getElementById('modal-title').textContent = `Document: ${doc.id}`;
            document.getElementById('modal-content').textContent = JSON.stringify(doc, null, 2);
            document.getElementById('modal-overlay').classList.add('active');
        }

        function closeModal(event) {
            if (!event || event.target.classList.contains('modal-overlay')) {
                document.getElementById('modal-overlay').classList.remove('active');
            }
        }

        // Close modal with Escape key
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>"##;

/// Get dashboard HTML with version injected
fn get_dashboard_html() -> String {
    DASHBOARD_HTML.replace("{VERSION}", env!("CARGO_PKG_VERSION"))
}

// ============================================================================
// API Response Types for Dashboard
// ============================================================================

/// Namespace info for API
#[derive(Debug, Serialize)]
pub struct NamespaceInfo {
    pub name: String,
    pub count: usize,
}

impl Clone for NamespaceInfo {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            count: self.count,
        }
    }
}

/// Namespaces list response
#[derive(Debug, Serialize)]
pub struct NamespacesResponse {
    pub namespaces: Vec<NamespaceInfo>,
    pub total: usize,
}

/// Overview response
#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub namespace_count: usize,
    pub total_documents: usize,
    pub db_path: String,
    pub embedding_provider: String,
}

/// Browse query params
#[derive(Debug, Deserialize)]
pub struct BrowseParams {
    #[serde(default = "default_browse_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_browse_limit() -> usize {
    50
}

/// Browse response
#[derive(Debug, Serialize)]
pub struct BrowseResponse {
    pub namespace: Option<String>,
    pub documents: Vec<SearchResultJson>,
    pub count: usize,
    pub offset: usize,
}

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
    /// Cached namespace list (refreshed in background for large DBs)
    pub cached_namespaces: Arc<RwLock<Option<Vec<NamespaceInfo>>>>,
    /// Per-namespace last activity timestamp (updated on upsert/index)
    pub namespace_activity: Arc<RwLock<HashMap<String, String>>>,
    /// Optional Bearer token for authenticating mutating requests
    pub auth_token: Option<String>,
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

/// Bearer token auth middleware for mutating endpoints.
/// If the server has an auth_token configured, requires `Authorization: Bearer <token>`.
/// Returns 401 if the token is missing or doesn't match.
async fn auth_middleware(
    State(state): State<HttpState>,
    request: Request,
    next: Next,
) -> impl IntoResponse {
    if let Some(ref expected) = state.auth_token {
        let auth_header = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                if token != expected.as_str() {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({"error": "missing or invalid auth token"})),
                    ));
                }
            }
            _ => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error": "missing or invalid auth token"})),
                ));
            }
        }
    }
    Ok(next.run(request).await)
}

/// HTTP server configuration passed to `create_router` and `start_server`
#[derive(Clone)]
pub struct HttpServerConfig {
    /// Bearer token for auth on mutating endpoints. None = no auth.
    pub auth_token: Option<String>,
    /// Allowed CORS origins. Empty = same-origin only (unless localhost).
    pub cors_origins: Vec<String>,
    /// Bind address. Defaults to 127.0.0.1.
    pub bind_address: IpAddr,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            auth_token: None,
            cors_origins: Vec::new(),
            bind_address: std::net::Ipv4Addr::LOCALHOST.into(),
        }
    }
}

/// Create the HTTP router
pub fn create_router(state: HttpState, config: &HttpServerConfig) -> Router {
    let is_localhost = config.bind_address.is_loopback();

    // CORS policy: permissive on localhost, restrictive otherwise
    let cors = if is_localhost && config.cors_origins.is_empty() {
        // Localhost with no explicit origins: permissive (safe since local only)
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    } else if config.cors_origins.is_empty() {
        // Non-localhost with no explicit origins: restrict to GET/POST, same-origin
        CorsLayer::new()
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
    } else {
        // Explicit origins configured
        let origins: Vec<HeaderValue> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
    };

    // Read-only routes (no auth required)
    let public_routes = Router::new()
        .route("/", get(dashboard_handler))
        .route("/api/discovery", get(discovery_handler))
        .route("/api/namespaces", get(namespaces_handler))
        .route("/api/overview", get(overview_handler))
        .route("/api/status", get(status_handler))
        .route("/api/browse", get(browse_all_handler))
        .route("/api/browse/", get(browse_all_handler))
        .route("/api/browse/{ns}", get(browse_handler))
        .route("/health", get(health_handler))
        .route("/search", post(search_handler))
        .route("/sse/search", get(sse_search_handler))
        .route("/cross-search", get(cross_search_handler))
        .route("/sse/cross-search", get(sse_cross_search_handler))
        .route("/sse/namespaces", get(sse_namespaces_handler))
        .route("/expand/{ns}/{id}", get(expand_handler))
        .route("/parent/{ns}/{id}", get(parent_handler))
        .route("/get/{ns}/{id}", get(get_handler));

    // Mutating routes (auth required when token is configured)
    let authed_routes = Router::new()
        .route("/refresh", post(refresh_handler))
        .route("/sse/optimize", post(sse_optimize_handler))
        .route("/upsert", post(upsert_handler))
        .route("/index", post(index_handler))
        .route("/delete/{ns}/{id}", post(delete_handler))
        .route("/ns/{namespace}", delete(purge_namespace_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    // MCP-over-SSE endpoints (auth required when token is configured)
    let mcp_routes = Router::new()
        .route("/mcp/", get(mcp_sse_handler))
        .route("/mcp/messages/", post(mcp_messages_handler))
        .route("/sse/", get(mcp_sse_handler))
        .route("/messages/", post(mcp_messages_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    public_routes
        .merge(authed_routes)
        .merge(mcp_routes)
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

// ============================================================================
// Dashboard & Browse API Handlers
// ============================================================================

/// Dashboard HTML endpoint (GET /)
async fn dashboard_handler() -> Html<String> {
    debug!("Dashboard: serving HTML");
    Html(get_dashboard_html())
}

/// List all namespaces with document counts (GET /api/namespaces)
/// Uses cached namespace list (refreshed in background every 5 minutes)
/// Falls back to "loading" state if cache not yet populated
async fn namespaces_handler(State(state): State<HttpState>) -> Json<NamespacesResponse> {
    // Try to use cached namespaces first (instant response)
    let cache = state.cached_namespaces.read().await;
    if let Some(ref namespaces) = *cache {
        let mut sorted = namespaces.clone();
        sorted.sort_by(|a, b| b.count.cmp(&a.count));
        let total = sorted.len();
        debug!(
            "API: /api/namespaces - returning {} cached namespaces",
            total
        );
        return Json(NamespacesResponse {
            namespaces: sorted,
            total,
        });
    }
    drop(cache);

    // Cache not ready yet - return loading indicator
    // Dashboard will show "loading" state and auto-refresh
    info!("API: /api/namespaces - cache not ready, background task loading...");
    Json(NamespacesResponse {
        namespaces: vec![],
        total: 0,
    })
}

/// Database overview (GET /api/overview)
/// Uses efficient stats() which only counts rows without loading data
async fn overview_handler(
    State(state): State<HttpState>,
) -> Result<Json<OverviewResponse>, (StatusCode, String)> {
    info!("API: /api/overview - fetching stats");

    // Use efficient stats() - only counts rows, doesn't load all data
    let stats = state.rag.storage().stats().await.map_err(|e| {
        error!("API: /api/overview - stats error: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    info!("API: /api/overview - {} documents", stats.row_count);

    // Note: namespace_count requires scanning, so we report 0 for efficiency
    // The namespaces endpoint provides the detailed breakdown
    Ok(Json(OverviewResponse {
        namespace_count: 0, // Use /api/namespaces for accurate count
        total_documents: stats.row_count,
        db_path: stats.db_path,
        embedding_provider: state.rag.mlx_connected_to(),
    }))
}

/// System status including cache state (GET /api/status)
/// Returns info about whether namespace cache is ready (for dashboard)
async fn status_handler(State(state): State<HttpState>) -> Json<serde_json::Value> {
    let cache = state.cached_namespaces.read().await;
    let cache_ready = cache.is_some();
    let namespace_count = cache.as_ref().map(|v| v.len()).unwrap_or(0);
    drop(cache);

    Json(json!({
        "cache_ready": cache_ready,
        "namespace_count": namespace_count,
        "hint": if !cache_ready {
            "Namespace cache loading... If this persists, run: rmcp-memex optimize"
        } else {
            "OK"
        }
    }))
}

/// Browse documents in namespace (GET /api/browse/:ns)
async fn browse_handler(
    State(state): State<HttpState>,
    Path(ns): Path<String>,
    Query(params): Query<BrowseParams>,
) -> Result<Json<BrowseResponse>, (StatusCode, String)> {
    info!(
        "API: /api/browse/{} - limit={}, offset={}",
        ns, params.limit, params.offset
    );

    let namespace = if ns.is_empty() {
        None
    } else {
        Some(ns.as_str())
    };

    // Use a zero embedding to get all docs (sorted by default order)
    let zero_embedding = vec![0.0_f32; 4096];
    let all_docs = state
        .rag
        .storage()
        .search_store(namespace, zero_embedding, params.limit + params.offset)
        .await
        .map_err(|e| {
            error!("API: /api/browse/{} - error: {}", ns, e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    // Apply offset and convert to SearchResultJson
    // ChromaDocument fields: id, namespace, embedding, metadata, document, layer (u8), parent_id, children_ids, keywords
    let documents: Vec<SearchResultJson> = all_docs
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
        .map(|doc| {
            let can_expand = !doc.children_ids.is_empty();
            let can_drill_up = doc.parent_id.is_some();
            let layer = SliceLayer::from_u8(doc.layer);
            SearchResultJson {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document, // ChromaDocument uses 'document' not 'text'
                score: 0.0,         // No score for browse (not a search result)
                metadata: doc.metadata,
                layer: layer.map(|l| l.name().to_string()),
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
                can_expand,
                can_drill_up,
            }
        })
        .collect();

    let count = documents.len();
    Ok(Json(BrowseResponse {
        namespace: if ns.is_empty() { None } else { Some(ns) },
        documents,
        count,
        offset: params.offset,
    }))
}

/// Browse all documents (no namespace filter) (GET /api/browse)
async fn browse_all_handler(
    State(state): State<HttpState>,
    Query(params): Query<BrowseParams>,
) -> Result<Json<BrowseResponse>, (StatusCode, String)> {
    info!(
        "API: /api/browse (all) - limit={}, offset={}",
        params.limit, params.offset
    );

    // Use a zero embedding to get documents (random order without real search)
    let zero_embedding = vec![0.0_f32; 4096];
    let all_docs = state
        .rag
        .storage()
        .search_store(None, zero_embedding, params.limit + params.offset)
        .await
        .map_err(|e| {
            error!("API: /api/browse (all) - error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    let documents: Vec<SearchResultJson> = all_docs
        .into_iter()
        .skip(params.offset)
        .take(params.limit)
        .map(|doc| {
            let can_expand = !doc.children_ids.is_empty();
            let can_drill_up = doc.parent_id.is_some();
            let layer = SliceLayer::from_u8(doc.layer);
            SearchResultJson {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document,
                score: 0.0,
                metadata: doc.metadata,
                layer: layer.map(|l| l.name().to_string()),
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
                can_expand,
                can_drill_up,
            }
        })
        .collect();

    let count = documents.len();
    Ok(Json(BrowseResponse {
        namespace: None,
        documents,
        count,
        offset: params.offset,
    }))
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

/// Minimal endpoint discovery — single source of truth for clients and dashboards
/// GET /api/discovery
///
/// Returns status, db info, and all namespaces with counts and last activity.
/// Replaces fragmented /api/namespaces + /api/overview + /api/status trio.
async fn discovery_handler(State(state): State<HttpState>) -> Json<serde_json::Value> {
    let cache = state.cached_namespaces.read().await;
    let activity = state.namespace_activity.read().await;

    let namespaces: Vec<serde_json::Value> = cache
        .as_ref()
        .map(|ns_list| {
            ns_list
                .iter()
                .map(|ns| {
                    json!({
                        "id": ns.name,
                        "count": ns.count,
                        "last_indexed_at": activity.get(&ns.name),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let total_documents: usize = cache
        .as_ref()
        .map(|ns| ns.iter().map(|n| n.count).sum())
        .unwrap_or(0);

    Json(json!({
        "status": if cache.is_some() { "ok" } else { "loading" },
        "version": env!("CARGO_PKG_VERSION"),
        "db_path": state.rag.storage().lance_path(),
        "embedding_provider": state.rag.mlx_connected_to(),
        "total_documents": total_documents,
        "namespaces": namespaces,
    }))
}

/// SSE streaming namespace listing with per-namespace summary
/// GET /sse/namespaces - streams each namespace with doc count, layer distribution, keywords
async fn sse_namespaces_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let start = std::time::Instant::now();

        yield Ok(Event::default()
            .event("start")
            .data(serde_json::json!({
                "status": "scanning_namespaces"
            }).to_string()));

        // Get namespace list
        let namespaces = match state.rag.storage().list_namespaces().await {
            Ok(ns) => ns,
            Err(e) => {
                yield Ok(Event::default()
                    .event("error")
                    .data(serde_json::json!({"error": e.to_string()}).to_string()));
                return;
            }
        };

        let total_namespaces = namespaces.len();
        let total_docs: usize = namespaces.iter().map(|(_, c)| *c).sum();

        yield Ok(Event::default()
            .event("overview")
            .data(serde_json::json!({
                "total_namespaces": total_namespaces,
                "total_documents": total_docs
            }).to_string()));

        // Stream per-namespace summary
        for (i, (ns_name, doc_count)) in namespaces.iter().enumerate() {
            // Get documents for this namespace to compute layer distribution + keywords
            let mut layer_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut all_keywords: Vec<String> = Vec::new();

            if let Ok(docs) = state.rag.storage().get_all_in_namespace(ns_name).await {
                for doc in &docs {
                    let layer_name = SliceLayer::from_u8(doc.layer)
                        .map(|l| l.name().to_string())
                        .unwrap_or_else(|| "flat".to_string());
                    *layer_counts.entry(layer_name).or_insert(0) += 1;

                    for kw in &doc.keywords {
                        if all_keywords.len() < 20 && !all_keywords.contains(kw) {
                            all_keywords.push(kw.clone());
                        }
                    }
                }
            }

            let ns_summary = serde_json::json!({
                "name": ns_name,
                "document_count": doc_count,
                "layers": layer_counts,
                "sample_keywords": all_keywords,
                "index": i,
            });

            yield Ok(Event::default()
                .event("namespace")
                .id(i.to_string())
                .data(ns_summary.to_string()));

            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        yield Ok(Event::default()
            .event("done")
            .data(serde_json::json!({
                "status": "complete",
                "total_namespaces": total_namespaces,
                "total_documents": total_docs,
                "elapsed_ms": start.elapsed().as_millis() as u64
            }).to_string()));
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// SSE streaming optimize endpoint - runs compact + prune with progress events
/// POST /sse/optimize - streams optimization progress and stats
async fn sse_optimize_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let start = std::time::Instant::now();

        // Pre-optimize stats
        let pre_stats = state.rag.storage().stats().await.ok();

        yield Ok(Event::default()
            .event("start")
            .data(serde_json::json!({
                "status": "starting_optimization",
                "db_path": state.rag.storage().lance_path(),
                "pre_row_count": pre_stats.as_ref().map(|s| s.row_count),
                "pre_version_count": pre_stats.as_ref().map(|s| s.version_count),
            }).to_string()));

        // Phase 1: Compact
        yield Ok(Event::default()
            .event("phase")
            .data(serde_json::json!({
                "phase": "compact",
                "status": "running",
                "description": "Merging small files into larger ones"
            }).to_string()));

        let compact_result = state.rag.storage().compact().await;

        match &compact_result {
            Ok(stats) => {
                yield Ok(Event::default()
                    .event("compact_done")
                    .data(serde_json::json!({
                        "phase": "compact",
                        "status": "complete",
                        "files_removed": stats.compaction.as_ref().map(|c| c.files_removed),
                        "files_added": stats.compaction.as_ref().map(|c| c.files_added),
                        "fragments_removed": stats.compaction.as_ref().map(|c| c.fragments_removed),
                        "fragments_added": stats.compaction.as_ref().map(|c| c.fragments_added),
                    }).to_string()));
            }
            Err(e) => {
                yield Ok(Event::default()
                    .event("compact_error")
                    .data(serde_json::json!({
                        "phase": "compact",
                        "status": "error",
                        "error": e.to_string()
                    }).to_string()));
            }
        }

        tokio::time::sleep(Duration::from_millis(10)).await;

        // Phase 2: Prune old versions
        yield Ok(Event::default()
            .event("phase")
            .data(serde_json::json!({
                "phase": "prune",
                "status": "running",
                "description": "Removing old versions (>7 days)"
            }).to_string()));

        let prune_result = state.rag.storage().cleanup(Some(7)).await;

        match &prune_result {
            Ok(stats) => {
                yield Ok(Event::default()
                    .event("prune_done")
                    .data(serde_json::json!({
                        "phase": "prune",
                        "status": "complete",
                        "old_versions": stats.prune.as_ref().map(|p| p.old_versions),
                        "bytes_removed": stats.prune.as_ref().map(|p| p.bytes_removed),
                    }).to_string()));
            }
            Err(e) => {
                yield Ok(Event::default()
                    .event("prune_error")
                    .data(serde_json::json!({
                        "phase": "prune",
                        "status": "error",
                        "error": e.to_string()
                    }).to_string()));
            }
        }

        // Post-optimize stats
        let post_stats = state.rag.storage().stats().await.ok();

        yield Ok(Event::default()
            .event("done")
            .data(serde_json::json!({
                "status": "complete",
                "post_row_count": post_stats.as_ref().map(|s| s.row_count),
                "post_version_count": post_stats.as_ref().map(|s| s.version_count),
                "compact_ok": compact_result.is_ok(),
                "prune_ok": prune_result.is_ok(),
                "elapsed_ms": start.elapsed().as_millis() as u64
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

    // Track namespace activity for discovery endpoint
    state
        .namespace_activity
        .write()
        .await
        .insert(req.namespace.clone(), chrono::Utc::now().to_rfc3339());

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

    // Track namespace activity for discovery endpoint
    state
        .namespace_activity
        .write()
        .await
        .insert(req.namespace.clone(), chrono::Utc::now().to_rfc3339());

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

    info!(
        "MCP SSE: New session {} (base_url: {})",
        session_id, base_url
    );

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
pub async fn start_server(
    rag: Arc<RAGPipeline>,
    port: u16,
    server_config: HttpServerConfig,
) -> anyhow::Result<()> {
    // Fallback base_url - actual URL is derived from Host header in mcp_sse_handler
    let base_url = format!("http://{}:{}", server_config.bind_address, port);
    let cached_namespaces = Arc::new(RwLock::new(None));

    // Log auth status
    if server_config.auth_token.is_some() {
        info!("HTTP auth: Bearer token required for mutating endpoints");
    } else {
        warn!(
            "WARNING: HTTP server running without auth token. Set MEMEX_AUTH_TOKEN or use --auth-token."
        );
    }

    // Warn if exposed on network without auth
    if !server_config.bind_address.is_loopback() && server_config.auth_token.is_none() {
        warn!(
            "WARNING: HTTP server exposed on network without auth token. Set MEMEX_AUTH_TOKEN or use --auth-token."
        );
    }

    let state = HttpState {
        rag: rag.clone(),
        mcp_sessions: Arc::new(McpSessionManager::new()),
        mcp_base_url: Arc::new(RwLock::new(base_url.clone())),
        cached_namespaces: cached_namespaces.clone(),
        namespace_activity: Arc::new(RwLock::new(HashMap::new())),
        auth_token: server_config.auth_token.clone(),
    };

    // Spawn background task to refresh namespace cache every 5 minutes
    let bg_rag = rag.clone();
    let bg_cache = cached_namespaces.clone();
    tokio::spawn(async move {
        // Initial load (with longer timeout for startup)
        info!("Background: Loading namespace cache (may take a while on large DB)...");
        match tokio::time::timeout(Duration::from_secs(120), bg_rag.storage().list_namespaces())
            .await
        {
            Ok(Ok(ns_list)) => {
                let namespaces: Vec<NamespaceInfo> = ns_list
                    .into_iter()
                    .map(|(name, count)| NamespaceInfo { name, count })
                    .collect();
                info!("Background: Cached {} namespaces", namespaces.len());
                *bg_cache.write().await = Some(namespaces);
            }
            Ok(Err(e)) => {
                // Database error (likely "too many open files" - needs optimize)
                warn!(
                    "Background: Namespace load FAILED: {} - run 'rmcp-memex optimize' to fix",
                    e
                );
            }
            Err(_) => {
                warn!("Background: Namespace load timed out (120s) - will retry");
            }
        }

        // Refresh every 5 minutes
        let mut interval = tokio::time::interval(Duration::from_secs(300));
        interval.tick().await; // Skip first immediate tick

        loop {
            interval.tick().await;
            debug!("Background: Refreshing namespace cache...");
            match tokio::time::timeout(Duration::from_secs(60), bg_rag.storage().list_namespaces())
                .await
            {
                Ok(Ok(ns_list)) => {
                    let namespaces: Vec<NamespaceInfo> = ns_list
                        .into_iter()
                        .map(|(name, count)| NamespaceInfo { name, count })
                        .collect();
                    info!("Background: Refreshed {} namespaces", namespaces.len());
                    *bg_cache.write().await = Some(namespaces);
                }
                Ok(Err(e)) => {
                    warn!(
                        "Background: Namespace refresh FAILED: {} - run 'rmcp-memex optimize'",
                        e
                    );
                }
                Err(_) => {
                    debug!("Background: Namespace refresh timed out");
                }
            }
        }
    });

    let app = create_router(state, &server_config);

    let addr = format!("{}:{}", server_config.bind_address, port);
    info!("HTTP/SSE server starting on http://{}", addr);
    info!("  Dashboard: http://{}/ (browse memories visually)", addr);
    info!("  Discovery: /api/discovery (canonical endpoint)");
    info!("  API: /api/namespaces, /api/overview, /api/browse/:ns");
    info!("  Search: /search, /sse/search, /cross-search");
    info!("  MCP-SSE: /sse/, /messages/");

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
