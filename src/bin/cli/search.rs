use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

use rmcp_memex::{
    BM25Config, EmbeddingClient, EmbeddingConfig, HybridConfig, HybridSearcher, RAGPipeline,
    SearchMode, SearchOptions, SliceLayer, StorageManager, path_utils,
};

#[allow(dead_code)]
fn parse_features(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Standard config discovery locations (in priority order)
#[allow(dead_code)]
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "~/.rmcp-servers/rmcp-memex/config.toml",
    "~/.config/rmcp-memex/config.toml",
    "~/.rmcp_servers/rmcp_memex/config.toml", // legacy underscore path
];

/// Discover config file from standard locations
#[allow(dead_code)]
fn discover_config() -> Option<String> {
    // 1. Environment variable takes priority
    if let Ok(path) = std::env::var("RMCP_MEMEX_CONFIG") {
        let expanded = shellexpand::tilde(&path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path);
        }
    }

    // 2. Check standard locations
    for path in CONFIG_SEARCH_PATHS {
        let expanded = shellexpand::tilde(path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path.to_string());
        }
    }

    None
}

#[allow(dead_code)]
fn load_file_config(path: &str) -> Result<FileConfig> {
    let (_canonical, contents) = path_utils::safe_read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot load config '{}': {}", path, e))?;
    toml::from_str(&contents).map_err(Into::into)
}

/// Load config from explicit path or discover from standard locations
#[allow(dead_code)]
fn load_or_discover_config(explicit_path: Option<&str>) -> Result<(FileConfig, Option<String>)> {
    // Explicit path takes priority
    if let Some(path) = explicit_path {
        return Ok((load_file_config(path)?, Some(path.to_string())));
    }

    // Try to discover config
    if let Some(discovered) = discover_config() {
        return Ok((load_file_config(&discovered)?, Some(discovered)));
    }

    // No config found - use defaults
    Ok((FileConfig::default(), None))
}

use crate::cli::config::*;
use crate::cli::formatting::*;
/// Check if auto-optimization should run and execute if needed
pub async fn check_and_maybe_optimize(
    storage: &StorageManager,
    maintenance_config: &Option<MaintenanceFileConfig>,
) -> Result<bool> {
    let config = match maintenance_config {
        Some(c) if c.auto_optimize => c,
        _ => return Ok(false), // Auto-optimize disabled
    };

    let stats = storage.stats().await?;

    if stats.version_count > config.version_threshold {
        eprintln!(
            "Auto-optimizing: {} versions exceed threshold {}",
            stats.version_count, config.version_threshold
        );
        storage.optimize().await?;

        // Also run cleanup if configured
        if let Some(days) = config.auto_cleanup_days {
            storage.cleanup(Some(days)).await?;
        }

        eprintln!("Auto-optimization complete");
        return Ok(true);
    }

    Ok(false)
}

/// Configuration for semantic search
pub struct SearchConfig<'a> {
    pub namespace: String,
    pub query: String,
    pub limit: usize,
    pub json_output: bool,
    pub db_path: String,
    pub layer_filter: Option<SliceLayer>,
    pub search_mode: SearchMode,
    pub embedding_config: &'a EmbeddingConfig,
}

/// Run semantic search within a namespace
pub async fn run_search(config: SearchConfig<'_>) -> Result<()> {
    let SearchConfig {
        namespace,
        query,
        limit,
        json_output,
        db_path,
        layer_filter,
        search_mode,
        embedding_config,
    } = config;
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);

    // Use hybrid search if mode is not pure vector
    if search_mode != SearchMode::Vector {
        // Create hybrid config with specified mode (read-only for CLI to avoid lock conflicts)
        let hybrid_config = HybridConfig {
            mode: search_mode,
            bm25: BM25Config {
                read_only: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let hybrid_searcher = HybridSearcher::new(storage, hybrid_config).await?;

        // Get query embedding
        let query_embedding = embedding_client.lock().await.embed(&query).await?;

        let results = hybrid_searcher
            .search(
                &query,
                query_embedding,
                Some(&namespace),
                limit,
                SearchOptions {
                    layer_filter,
                    project_filter: None,
                },
            )
            .await?;

        if json_output {
            let json = json_hybrid_search_results(
                &query,
                Some(&namespace),
                &results,
                layer_filter,
                search_mode,
            );
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            display_hybrid_search_results(
                &query,
                Some(&namespace),
                &results,
                layer_filter,
                search_mode,
            );
        }
    } else {
        // Legacy vector-only search
        let rag = RAGPipeline::new(embedding_client, storage).await?;
        let results = rag
            .memory_search_with_layer(&namespace, &query, limit, layer_filter)
            .await?;

        if json_output {
            let json = json_search_results(&query, Some(&namespace), &results, layer_filter);
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            display_search_results(&query, Some(&namespace), &results, layer_filter);
        }
    }

    Ok(())
}

/// Expand a slice to get its children (drill down in onion hierarchy)
pub async fn run_expand(
    namespace: String,
    id: String,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    let results = rag.expand_result(&namespace, &id).await?;

    if json_output {
        let json = serde_json::json!({
            "parent_id": id,
            "namespace": namespace,
            "children_count": results.len(),
            "children": results.iter().map(|r| serde_json::json!({
                "id": r.id,
                "layer": r.layer.map(|l| l.name()),
                "text": r.text,
                "keywords": r.keywords,
                "parent_id": r.parent_id,
                "children_ids": r.children_ids,
            })).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n-> Children of slice \"{id}\" in [{namespace}]\n");

        if results.is_empty() {
            eprintln!("No children found (this may be a leaf/outer slice).");
        } else {
            for (i, result) in results.iter().enumerate() {
                let layer_str = result.layer.map(|l| l.name()).unwrap_or("flat");
                let preview: String = result
                    .text
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ");
                let ellipsis = if result.text.len() > 100 { "..." } else { "" };

                eprintln!("{}. [{}] {}", i + 1, layer_str, result.id);
                eprintln!("   \"{}{ellipsis}\"", preview);
                if !result.keywords.is_empty() {
                    eprintln!("   Keywords: {}", result.keywords.join(", "));
                }
                eprintln!();
            }
        }
    }

    Ok(())
}

/// Get a specific chunk by namespace and ID
pub async fn run_get(
    namespace: String,
    id: String,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    match rag.lookup_memory(&namespace, &id).await? {
        Some(result) => {
            if json_output {
                let json = serde_json::json!({
                    "found": true,
                    "id": result.id,
                    "namespace": result.namespace,
                    "text": result.text,
                    "metadata": result.metadata
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                eprintln!("\n-> Found chunk in [{namespace}]\n");
                eprintln!("ID: {}", result.id);
                eprintln!("Namespace: {}", result.namespace);
                if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
                    eprintln!("Metadata: {}", result.metadata);
                }
                eprintln!("\n--- Content ---\n");
                println!("{}", result.text);
            }
        }
        None => {
            if json_output {
                let json = serde_json::json!({
                    "found": false,
                    "namespace": namespace,
                    "id": id
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                eprintln!("Chunk '{}' not found in namespace '{}'", id, namespace);
            }
        }
    }

    Ok(())
}

/// Run RAG search (optionally across all namespaces)
pub async fn run_rag_search(
    query: String,
    limit: usize,
    namespace: Option<String>,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    let results = rag
        .search_inner(namespace.as_deref(), &query, limit)
        .await?;

    if json_output {
        let json = json_search_results(&query, namespace.as_deref(), &results, None);
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        display_search_results(&query, namespace.as_deref(), &results, None);
    }

    Ok(())
}

/// List all namespaces with optional stats
pub async fn run_list_namespaces(stats: bool, json_output: bool, db_path: String) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // We need to query the LanceDB to get unique namespaces
    // This requires querying all documents and extracting unique namespaces
    let storage = Arc::new(storage);

    // Use a dummy embedding to get all documents (we'll use a zero vector)
    // This is a workaround since LanceDB doesn't have a direct "list all" method
    // We search with a large limit to get namespace statistics
    let all_docs = storage.all_documents(None, 10000).await?;

    // Collect unique namespaces with counts
    let mut namespace_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for doc in &all_docs {
        *namespace_counts.entry(doc.namespace.clone()).or_insert(0) += 1;
    }

    let mut namespaces: Vec<_> = namespace_counts.into_iter().collect();
    namespaces.sort_by(|a, b| a.0.cmp(&b.0));

    if json_output {
        let json = if stats {
            serde_json::json!({
                "namespaces": namespaces.iter().map(|(ns, count)| serde_json::json!({
                    "name": ns,
                    "document_count": count
                })).collect::<Vec<_>>()
            })
        } else {
            serde_json::json!({
                "namespaces": namespaces.iter().map(|(ns, _)| ns).collect::<Vec<_>>()
            })
        };
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n-> Namespaces in {}\n", storage.lance_path());

        if namespaces.is_empty() {
            eprintln!("No namespaces found (database may be empty).");
        } else {
            for (ns, count) in &namespaces {
                if stats {
                    eprintln!("  {} ({} documents)", ns, count);
                } else {
                    eprintln!("  {}", ns);
                }
            }
            eprintln!();
            eprintln!("Total: {} namespace(s)", namespaces.len());
        }
    }

    Ok(())
}

/// Cross-search result with namespace information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSearchResult {
    pub id: String,
    pub namespace: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub keywords: Vec<String>,
}

/// Search across all namespaces, merge results by score
pub async fn run_cross_search(
    query: String,
    limit_per_ns: usize,
    total_limit: usize,
    mode: String,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);

    // First, get list of all namespaces
    let all_docs = storage.all_documents(None, 10000).await?;

    let mut namespace_set: HashSet<String> = HashSet::new();
    for doc in &all_docs {
        namespace_set.insert(doc.namespace.clone());
    }

    let namespaces: Vec<String> = namespace_set.into_iter().collect();

    if namespaces.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::json!({ "results": [], "total": 0, "namespaces_searched": 0 })
            );
        } else {
            eprintln!("No namespaces found in database.");
        }
        return Ok(());
    }

    if !json_output {
        eprintln!(
            "Searching {} namespaces for: \"{}\"",
            namespaces.len(),
            query
        );
        eprintln!(
            "Mode: {}, limit per namespace: {}, total limit: {}",
            mode, limit_per_ns, total_limit
        );
        eprintln!();
    }

    // Parse search mode and configure hybrid searcher
    let search_mode = match mode.as_str() {
        "vector" => SearchMode::Vector,
        "keyword" | "bm25" => SearchMode::Keyword,
        _ => SearchMode::Hybrid,
    };

    // Create hybrid config with the specified mode (read-only for CLI)
    let hybrid_config = HybridConfig {
        mode: search_mode,
        bm25: BM25Config {
            read_only: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let hybrid_searcher = HybridSearcher::new(storage.clone(), hybrid_config).await?;

    // Embed the query once for all namespaces
    let query_embedding = embedding_client.lock().await.embed(&query).await?;

    // Search each namespace and collect results
    let mut all_results: Vec<CrossSearchResult> = Vec::new();

    for ns in &namespaces {
        let ns_results = hybrid_searcher
            .search(
                &query,
                query_embedding.clone(),
                Some(ns.as_str()),
                limit_per_ns,
                SearchOptions::default(),
            )
            .await?;

        for r in ns_results {
            all_results.push(CrossSearchResult {
                id: r.id,
                namespace: r.namespace,
                text: r.document,
                score: r.combined_score,
                metadata: r.metadata,
                layer: r.layer.map(|l| l.to_string()),
                keywords: r.keywords,
            });
        }
    }

    // Sort by score descending
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to total_limit
    all_results.truncate(total_limit);

    if json_output {
        let output = serde_json::json!({
            "query": query,
            "mode": mode,
            "namespaces_searched": namespaces.len(),
            "total_results": all_results.len(),
            "results": all_results
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!(
            "Found {} results across {} namespaces:\n",
            all_results.len(),
            namespaces.len()
        );

        for (idx, r) in all_results.iter().enumerate() {
            eprintln!(
                "{}. [{}] {} (score: {:.4})",
                idx + 1,
                r.namespace,
                &r.id,
                r.score
            );
            if let Some(ref layer) = r.layer {
                eprintln!("   Layer: {}", layer);
            }
            if !r.keywords.is_empty() {
                eprintln!("   Keywords: {}", r.keywords.join(", "));
            }
            // Truncate text for display
            let preview = if r.text.len() > 200 {
                format!("{}...", &r.text[..200])
            } else {
                r.text.clone()
            };
            eprintln!("   {}\n", preview.replace('\n', " "));
        }
    }

    Ok(())
}
