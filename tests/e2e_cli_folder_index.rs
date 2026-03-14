//! Self-contained CLI E2E coverage for folder indexing and search.
//!
//! This test uses real sample documents from ~/.ai-contexters/mlx-embeddings
//! when available, but keeps a fallback corpus so CI remains deterministic.

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use rmcp_memex::StorageManager;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};
use tempfile::TempDir;
use tokio::{net::TcpListener, task::JoinHandle};

const REQUIRED_DIMENSION: usize = 4096;
const TEST_NAMESPACE: &str = "e2e-cli-folder";
const SEARCH_QUERY: &str = "Apple Silicon local AI inference batch processing";
const FALLBACK_README: &str = r#"
# MLX Batch Server

High-performance local AI inference server for Apple Silicon with batch processing,
OpenAI-compatible embeddings, and practical developer ergonomics. The server is
designed for local AI workloads on Apple Silicon where throughput, latency, and
observability all matter at the same time. This README explains how batch
processing works, how the OpenAI Responses API is exposed, and how operators can
keep the service stable under load.

The project focuses on Apple Silicon acceleration, local AI inference, concurrent
batch processing, and compatibility with existing OpenAI client code. A real
deployment usually wants strong defaults, readable logs, and a fast path for
local experimentation. Those themes appear throughout the documentation because
the product goal is clear: ship a production-grade local AI inference server that
still feels simple for developers who just want embeddings, responses, and a
predictable operational model.

Use cases include local embeddings for document search, Responses API flows for
tool-using agents, and batch processing for many concurrent requests. The server
can expose OpenAI-compatible routes while still being tuned for Apple Silicon and
the practical realities of laptop and workstation inference.
"#;
const FALLBACK_SEARCH_TOOLS: &str = r#"
# Search Tools Comparison

This document compares search tools for local AI systems. It evaluates keyword
search, vector search, hybrid retrieval, and response ranking across several
practical workflows. The emphasis is on how teams choose between semantic search,
lexical search, and mixed strategies when working with technical documentation,
code snippets, and operational runbooks.

Hybrid retrieval often wins in production because it balances semantic recall
with deterministic keyword anchors. That matters when users search for error
messages, API names, or precise command flags. Evaluation should include latency,
index freshness, retrieval explainability, and resilience under partial data
quality issues. Teams should also compare onboarding cost, operator visibility,
and the impact of chunk size on search quality.
"#;
const FALLBACK_CODEX_CONTEXT: &str = r#"
[project: mlx-embeddings | agent: codex | date: 2026-02-22]

The user wanted to know which tests were real end-to-end runs and which ones
were mock-driven. We inspected the fixtures, the server lifecycle, and the
responses chain behavior to determine whether the tool-use and streaming paths
were actually being exercised with loaded models. The conclusion was that some
tests used real inference while the Responses chain and SSE coverage still needed
an explicit live-server pass.

That distinction matters for shipping quality because green tests can hide a
product surface that is only partially real. The next step was to run targeted
server tests with a real model, gather the exact pass/fail map, and reduce log
noise for expected 404 scenarios in negative-path tests.
"#;

#[derive(Clone)]
struct MockEmbeddingState {
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct MockEmbeddingRequest {
    input: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MockEmbeddingResponse {
    data: Vec<MockEmbeddingData>,
}

#[derive(Debug, Serialize)]
struct MockEmbeddingData {
    embedding: Vec<f32>,
}

struct MockEmbeddingServer {
    base_url: String,
    handle: JoinHandle<()>,
}

impl Drop for MockEmbeddingServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct SeededCorpus {
    root: PathBuf,
    file_count: usize,
    expected_hit_suffix: String,
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cli_indexes_folder_samples_with_chunking_and_rag_search() {
    let server = start_mock_embedding_server().await;
    let tmp = TempDir::new().expect("failed to create temp dir");
    let corpus = seed_corpus(tmp.path()).expect("failed to create sample corpus");
    let db_path = tmp.path().join("lancedb");
    let config_path = write_config(&server.base_url).expect("failed to write config");

    let index_output = run_cli(
        env!("CARGO_BIN_EXE_rmmx"),
        [
            "--config",
            config_path.to_str().unwrap(),
            "--db-path",
            db_path.to_str().unwrap(),
            "--allowed-paths",
            corpus.root.to_str().unwrap(),
            "--allowed-paths",
            tmp.path().to_str().unwrap(),
            "index",
            corpus.root.to_str().unwrap(),
            "--namespace",
            TEST_NAMESPACE,
            "--recursive",
            "--preprocess",
            "--slice-mode",
            "flat",
            "--parallel",
            "1",
        ],
    );
    assert!(
        index_output.status.success(),
        "index command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&index_output.stdout),
        String::from_utf8_lossy(&index_output.stderr)
    );
    let index_stderr = String::from_utf8_lossy(&index_output.stderr);
    assert!(
        index_stderr.contains("Indexing complete"),
        "expected CLI summary in stderr, got:\n{}",
        index_stderr
    );

    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("failed to open LanceDB");
    let docs = storage
        .search_store(
            Some(TEST_NAMESPACE),
            vec![0.0_f32; REQUIRED_DIMENSION],
            10_000,
        )
        .await
        .expect("failed to read indexed documents");

    assert!(
        !docs.is_empty(),
        "expected indexed documents for namespace {}",
        TEST_NAMESPACE
    );

    let indexed_paths: HashSet<&str> = docs
        .iter()
        .filter_map(|doc| doc.metadata.get("path").and_then(|value| value.as_str()))
        .collect();
    assert_eq!(
        indexed_paths.len(),
        corpus.file_count,
        "expected all source files to be indexed exactly once at file level"
    );
    assert!(
        docs.len() > corpus.file_count,
        "expected flat chunking to produce more stored chunks than source files"
    );
    assert!(
        docs.iter().any(|doc| {
            doc.metadata
                .get("chunk_index")
                .and_then(|value| value.as_u64())
                .is_some()
        }),
        "expected chunk metadata to include chunk_index"
    );
    let max_total_chunks = docs
        .iter()
        .filter_map(|doc| {
            doc.metadata
                .get("total_chunks")
                .and_then(|value| value.as_u64())
        })
        .max()
        .unwrap_or(0);
    assert!(
        max_total_chunks > 1,
        "expected at least one multi-chunk document, got max total_chunks={}",
        max_total_chunks
    );

    let search_output = run_cli(
        env!("CARGO_BIN_EXE_rmmx"),
        [
            "--config",
            config_path.to_str().unwrap(),
            "--db-path",
            db_path.to_str().unwrap(),
            "--allowed-paths",
            tmp.path().to_str().unwrap(),
            "rag-search",
            "--query",
            SEARCH_QUERY,
            "--namespace",
            TEST_NAMESPACE,
            "--limit",
            "5",
            "--json",
        ],
    );
    assert!(
        search_output.status.success(),
        "rag-search failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&search_output.stdout),
        String::from_utf8_lossy(&search_output.stderr)
    );

    let response: Value =
        serde_json::from_slice(&search_output.stdout).expect("rag-search should emit valid JSON");
    let results = response["results"]
        .as_array()
        .expect("rag-search JSON should contain results");
    assert!(!results.is_empty(), "expected at least one search hit");
    assert!(
        results.iter().any(|result| {
            result
                .get("metadata")
                .and_then(|metadata| metadata.get("path"))
                .and_then(|value| value.as_str())
                .map(|path| path.ends_with(&corpus.expected_hit_suffix))
                .unwrap_or(false)
                || result
                    .get("text")
                    .and_then(|value| value.as_str())
                    .map(|text| text.contains("Apple Silicon"))
                    .unwrap_or(false)
        }),
        "expected search hits anchored in the README sample\nstdout:\n{}",
        String::from_utf8_lossy(&search_output.stdout)
    );
}

async fn start_mock_embedding_server() -> MockEmbeddingServer {
    let app = Router::new()
        .route("/v1/models", get(mock_models))
        .route("/v1/embeddings", post(mock_embeddings))
        .with_state(MockEmbeddingState {
            dimension: REQUIRED_DIMENSION,
        });

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind mock embedding server");
    let address = listener
        .local_addr()
        .expect("failed to read mock embedding server address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock embedding server failed");
    });

    MockEmbeddingServer {
        base_url: format!("http://{}", address),
        handle,
    }
}

async fn mock_models() -> Json<Value> {
    Json(json!({
        "data": [
            { "id": "mock-embedder" }
        ]
    }))
}

async fn mock_embeddings(
    State(state): State<MockEmbeddingState>,
    Json(request): Json<MockEmbeddingRequest>,
) -> Json<MockEmbeddingResponse> {
    let data = request
        .input
        .into_iter()
        .map(|text| MockEmbeddingData {
            embedding: hashed_embedding(&text, state.dimension),
        })
        .collect();

    Json(MockEmbeddingResponse { data })
}

fn hashed_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0_f32; dimension];

    for token in text
        .to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| token.len() >= 3)
    {
        let idx = fnv1a(token.as_bytes()) % dimension;
        embedding[idx] += 1.0;
    }

    if embedding.iter().all(|value| *value == 0.0) {
        embedding[0] = 1.0;
    }

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for value in &mut embedding {
        *value /= norm;
    }

    embedding
}

fn fnv1a(bytes: &[u8]) -> usize {
    let mut hash = 0x811c9dc5_u32;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    hash as usize
}

fn seed_corpus(base: &Path) -> std::io::Result<SeededCorpus> {
    let corpus_root = base.join("mlx-embeddings-samples");
    let candidates = [
        (
            source_root().join("README.md"),
            PathBuf::from("README.md"),
            Some("README.md"),
        ),
        (
            source_root().join("docs/search-tools-comparison.md"),
            PathBuf::from("docs/search-tools-comparison.md"),
            None,
        ),
        (
            source_root().join("2026-02-22/040553_codex-001.md"),
            PathBuf::from("2026-02-22/040553_codex-001.md"),
            None,
        ),
    ];

    let mut copied = 0usize;
    let mut expected_hit_suffix = None;

    for (source, relative_target, hit_suffix) in candidates {
        if source.exists() {
            let target = corpus_root.join(&relative_target);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source, &target)?;
            copied += 1;
            if expected_hit_suffix.is_none() {
                expected_hit_suffix = hit_suffix.map(str::to_string);
            }
        }
    }

    if copied == 0 {
        let fallback_docs = vec![
            (PathBuf::from("README.md"), FALLBACK_README),
            (
                PathBuf::from("docs/search-tools-comparison.md"),
                FALLBACK_SEARCH_TOOLS,
            ),
            (
                PathBuf::from("2026-02-22/040553_codex-001.md"),
                FALLBACK_CODEX_CONTEXT,
            ),
        ];
        let fallback_count = fallback_docs.len();

        for (relative_target, contents) in &fallback_docs {
            let target = corpus_root.join(relative_target);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(target, contents)?;
        }

        copied = fallback_count;
        expected_hit_suffix = Some("README.md".to_string());
    }

    Ok(SeededCorpus {
        root: corpus_root,
        file_count: copied,
        expected_hit_suffix: expected_hit_suffix.unwrap_or_else(|| "README.md".to_string()),
    })
}

fn source_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    Path::new(&home)
        .join(".ai-contexters")
        .join("mlx-embeddings")
}

fn write_config(base_url: &str) -> std::io::Result<PathBuf> {
    let config_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/e2e-configs");
    fs::create_dir_all(&config_dir)?;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let config_path = config_dir.join(format!("mock-embeddings-{unique}.toml"));
    let config = format!(
        "[embeddings]\nrequired_dimension = {REQUIRED_DIMENSION}\nmax_batch_chars = 32000\nmax_batch_items = 16\n\n[[embeddings.providers]]\nname = \"mock-e2e\"\nbase_url = \"{base_url}\"\nmodel = \"mock-embedder\"\npriority = 1\nendpoint = \"/v1/embeddings\"\n"
    );
    fs::write(&config_path, config)?;
    Ok(config_path)
}

fn run_cli<const N: usize>(binary: &str, args: [&str; N]) -> Output {
    Command::new(binary)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .expect("failed to run CLI")
}
