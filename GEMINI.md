# rmcp-memex

RAG/Memory MCP Server with LanceDB vector storage for AI agents.

## Project Overview

`rmcp-memex` is a comprehensive Model Context Protocol (MCP) server designed to provide Retrieval-Augmented Generation (RAG) and long-term memory capabilities to AI agents. It leverages LanceDB for efficient vector storage and Tantivy for BM25 keyword search, offering a hybrid search approach.

Key features include:
- **Hybrid Search:** Combines semantic vector search with BM25 keyword search for high relevance.
- **Onion Slice Architecture:** Hierarchical chunking (Outer, Middle, Inner, Core) to balance context window usage and information density.
- **Namespace Isolation:** Data is organized into namespaces with optional token-based security.
- **Multi-Agent Access:** specific HTTP/SSE server mode to allow multiple agents to access the same LanceDB instance (which normally locks exclusively).
- **Preprocessing:** Built-in noise filtering for conversation exports.

## Architecture

The project is structured as both a Rust library (`rmcp_memex`) and a CLI binary (`rmcp-memex`).

- **Core Library (`src/lib.rs`):** Exports the main logic for embeddings, storage, RAG pipeline, and MCP handlers.
- **CLI (`src/bin/rmcp_memex.rs`):** Provides commands for serving the MCP, indexing files, searching, and maintenance.
- **Storage Layer:** Uses LanceDB for vectors and Tantivy for text indexing.
- **Embeddings:** Supports Ollama (recommended), OpenAI-compatible APIs, and MLX (via bridge).

## Building and Running

The project uses `cargo` for building and a `Makefile` for convenience commands.

### Prerequisites
- Rust toolchain (stable)
- `protoc` (protobuf compiler) may be required for some dependencies.

### Common Commands

*   **Build Release Binary:**
    ```bash
    cargo build --release
    # OR
    make build
    ```

*   **Install to `~/.cargo/bin`:**
    ```bash
    make install
    ```

*   **Run Development Server:**
    ```bash
    make dev
    ```

*   **Run Tests:**
    ```bash
    cargo test
    ```

### Service Management (macOS/launchd)

The `Makefile` includes targets for managing the server as a background service:
- `make start`: Start the service.
- `make stop`: Stop the service.
- `make restart`: Restart the service.
- `make status`: Check service status.
- `make logs`: Tail server logs.

### RAM Disk Mode (Performance)

For high-performance setups (e.g., on machines with ample RAM), the project supports running the database from a RAM disk:
- `make ramdisk-up`: Create RAM disk, copy DB, and start service.
- `make ramdisk-down`: Sync to disk, unmount, and stop.
- `make snapshot`: Backup RAM disk content to persistent storage.

## Configuration

Configuration is handled via TOML files (e.g., `~/.rmcp-servers/rmcp-memex/config.toml`) or environment variables.

Key Env Vars:
- `OLLAMA_BASE_URL`: URL for Ollama (default: `http://localhost:11434`)
- `MEMEX_DB_PATH`: Path to LanceDB storage.

## Development Conventions

- **Code Style:** Standard Rust formatting (`cargo fmt`).
- **Error Handling:** Uses `anyhow::Result` for application-level errors.
- **Logging:** Uses `tracing` library.
- **Testing:** Unit tests in `tests` modules within files, integration tests in `tests/` directory.
- **Architecture:** "Onion Architecture" concept applied to data chunking (Outer -> Core).
