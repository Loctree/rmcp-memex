use anyhow::Result;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

use rmcp_memex::{
    BM25Config, BM25Index, EmbeddingClient, EmbeddingConfig, HealthChecker, RAGPipeline,
    SliceLayer, StorageManager, inspect_cross_store_recovery, path_utils,
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
/// Namespace overview stats
#[derive(Debug, Clone, serde::Serialize)]
pub struct NamespaceStats {
    pub name: String,
    pub total_chunks: usize,
    pub layer_counts: std::collections::HashMap<String, usize>,
    top_keywords: Vec<(String, usize)>,
    pub has_timestamps: bool,
    pub earliest_indexed: Option<String>,
    pub latest_indexed: Option<String>,
}

/// Run overview command - quick stats and health check
pub async fn run_overview(
    namespace: Option<String>,
    json_output: bool,
    db_path: String,
) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;
    let storage = Arc::new(storage);

    // Use a zero embedding to get all documents
    let all_docs = storage.all_documents(namespace.as_deref(), 100000).await?;

    if all_docs.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "empty",
                    "message": "No documents found",
                    "namespace": namespace,
                    "db_path": db_path
                }))?
            );
        } else {
            eprintln!("\n-> Overview for {}\n", storage.lance_path());
            if let Some(ns) = &namespace {
                eprintln!("No documents found in namespace '{}'", ns);
            } else {
                eprintln!("Database is empty. Use 'rmcp-memex index' to add documents.");
            }
        }
        return Ok(());
    }

    // Group by namespace
    let mut by_namespace: std::collections::HashMap<String, Vec<_>> =
        std::collections::HashMap::new();
    for doc in &all_docs {
        by_namespace
            .entry(doc.namespace.clone())
            .or_default()
            .push(doc);
    }

    let mut stats_list: Vec<NamespaceStats> = Vec::new();

    for (ns_name, docs) in &by_namespace {
        // Count layers
        let mut layer_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for doc in docs {
            let layer_name = match doc.layer {
                1 => "outer",
                2 => "middle",
                3 => "inner",
                4 => "core",
                _ => "flat",
            };
            *layer_counts.entry(layer_name.to_string()).or_insert(0) += 1;
        }

        // Collect all keywords and count frequency
        let mut keyword_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for doc in docs {
            for kw in &doc.keywords {
                *keyword_counts.entry(kw.clone()).or_insert(0) += 1;
            }
        }
        let mut top_keywords: Vec<_> = keyword_counts.into_iter().collect();
        top_keywords.sort_by_key(|b| std::cmp::Reverse(b.1));
        let top_keywords: Vec<(String, usize)> = top_keywords.into_iter().take(10).collect();

        // Check for timestamps in metadata (look for common timestamp patterns)
        let has_timestamps = docs.iter().any(|d| {
            let meta_str = d.metadata.to_string();
            meta_str.contains("timestamp")
                || meta_str.contains("created_at")
                || meta_str.contains("indexed_at")
                || meta_str.contains("date")
        });

        // Try to extract date range from metadata
        let mut dates: Vec<String> = Vec::new();
        for doc in docs {
            if let Some(obj) = doc.metadata.as_object() {
                for (k, v) in obj {
                    if (k.contains("date") || k.contains("timestamp") || k.contains("time"))
                        && let Some(s) = v.as_str()
                    {
                        dates.push(s.to_string());
                    }
                }
            }
        }
        dates.sort();

        stats_list.push(NamespaceStats {
            name: ns_name.clone(),
            total_chunks: docs.len(),
            layer_counts,
            top_keywords,
            has_timestamps,
            earliest_indexed: dates.first().cloned(),
            latest_indexed: dates.last().cloned(),
        });
    }

    stats_list.sort_by(|a, b| a.name.cmp(&b.name));

    if json_output {
        let json = serde_json::json!({
            "db_path": db_path,
            "total_chunks": all_docs.len(),
            "namespace_count": stats_list.len(),
            "namespaces": stats_list
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n=== RMCP-MEMEX OVERVIEW ===\n");
        eprintln!("Database: {}", db_path);
        eprintln!("Total chunks: {}", all_docs.len());
        eprintln!("Namespaces: {}\n", stats_list.len());

        for stats in &stats_list {
            eprintln!("--- {} ---", stats.name);
            eprintln!("  Chunks: {}", stats.total_chunks);

            // Layer breakdown
            if !stats.layer_counts.is_empty() {
                let layer_str: Vec<String> = stats
                    .layer_counts
                    .iter()
                    .map(|(k, v)| format!("{}:{}", k, v))
                    .collect();
                eprintln!("  Layers: {}", layer_str.join(", "));
            }

            // Top keywords
            if !stats.top_keywords.is_empty() {
                let kw_str: Vec<String> = stats
                    .top_keywords
                    .iter()
                    .take(5)
                    .map(|(k, v)| format!("{}({})", k, v))
                    .collect();
                eprintln!("  Top topics: {}", kw_str.join(", "));
            }

            // Date range
            if let (Some(earliest), Some(latest)) = (&stats.earliest_indexed, &stats.latest_indexed)
            {
                if earliest != latest {
                    eprintln!("  Date range: {} -> {}", earliest, latest);
                } else {
                    eprintln!("  Date: {}", earliest);
                }
            }

            // Timestamps warning
            if !stats.has_timestamps {
                eprintln!("  [!] No timestamp metadata found");
            }

            eprintln!();
        }

        eprintln!("Tip: Use 'rmcp-memex search -n <namespace> -q <query>' to search");
        eprintln!("     Use 'rmcp-memex dive -n <namespace> -q <query>' for deep exploration");
    }

    Ok(())
}

/// Health check result for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct HealthReport {
    pub database: DatabaseHealth,
    pub embedder: Option<EmbedderHealth>,
    pub namespaces: Vec<NamespaceHealth>,
    pub cross_store: Option<CrossStoreHealth>,
    pub recommendations: Vec<String>,
    pub overall_status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseHealth {
    pub path: String,
    pub status: String,
    pub row_count: usize,
    pub version_count: usize,
    pub size_estimate_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedderHealth {
    pub provider: Option<String>,
    pub status: String,
    pub dimension: Option<usize>,
    pub dimension_match: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamespaceHealth {
    pub name: String,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossStoreHealth {
    pub recovery_dir: String,
    pub status: String,
    pub pending_batches: usize,
    pub divergent_batches: usize,
    pub rolled_back_batches: usize,
    pub stale_batches: usize,
    pub missing_bm25_documents: usize,
    pub missing_lance_documents: usize,
}

/// Run health check command
pub async fn run_health(
    db_path: String,
    embedding_config: &EmbeddingConfig,
    config_path_display: Option<String>,
    quick: bool,
    json_output: bool,
) -> Result<()> {
    let mut recommendations = Vec::new();
    let mut overall_ok = true;

    // 1. Database health
    let db_health = match StorageManager::new_lance_only(&db_path).await {
        Ok(storage) => {
            let stats = storage.stats().await.unwrap_or(rmcp_memex::TableStats {
                row_count: 0,
                version_count: 0,
                table_name: "memories".to_string(),
                db_path: db_path.clone(),
            });

            // Estimate size: ~4KB per row (embedding + metadata)
            let size_mb = (stats.row_count as f64 * 4.0) / 1024.0;

            // Check version count threshold
            if stats.version_count > 50 {
                recommendations.push(format!(
                    "Run 'rmcp-memex optimize' ({} versions accumulated)",
                    stats.version_count
                ));
            }

            DatabaseHealth {
                path: db_path.clone(),
                status: "OK".to_string(),
                row_count: stats.row_count,
                version_count: stats.version_count,
                size_estimate_mb: size_mb,
            }
        }
        Err(e) => {
            overall_ok = false;
            recommendations.push(format!("Database error: {}", e));
            DatabaseHealth {
                path: db_path.clone(),
                status: format!("ERROR: {}", e),
                row_count: 0,
                version_count: 0,
                size_estimate_mb: 0.0,
            }
        }
    };

    // 2. Embedder health (skip if --quick)
    let embedder_health = if quick {
        None
    } else {
        let checker = HealthChecker::new();
        let result = checker.run_all(embedding_config, &db_path).await;

        let provider = result.connected_provider.clone();
        let dimension = result.verified_dimension;
        let dim_ok = dimension
            .map(|d| d == embedding_config.required_dimension)
            .unwrap_or(false);

        let status = if result.all_passed() {
            "OK".to_string()
        } else {
            overall_ok = false;
            let failures: Vec<_> = result
                .items
                .iter()
                .filter(|i| i.status.is_fail())
                .map(|i| i.name.clone())
                .collect();
            if provider.is_none() {
                recommendations.push(
                    "Embedder unreachable - check if embedding server is running".to_string(),
                );
            }
            format!("FAILED: {}", failures.join(", "))
        };

        Some(EmbedderHealth {
            provider,
            status,
            dimension,
            dimension_match: dim_ok,
        })
    };

    // 3. Namespace stats
    let namespaces = if let Ok(storage) = StorageManager::new_lance_only(&db_path).await {
        storage
            .list_namespaces()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|(name, count)| NamespaceHealth {
                name,
                chunk_count: count,
            })
            .collect()
    } else {
        vec![]
    };

    let cross_store = match StorageManager::new_lance_only(&db_path).await {
        Ok(storage) => match BM25Index::new(&BM25Config::default().with_read_only(true)) {
            Ok(bm25) => {
                let recovery = inspect_cross_store_recovery(&storage, &bm25, None).await?;
                let status = if recovery.pending_batches == 0 {
                    "OK".to_string()
                } else if recovery.divergent_batches > 0 {
                    overall_ok = false;
                    recommendations.push(
                        "Run 'rmcp-memex repair-writes --execute' to reconcile Lance/BM25 divergence"
                            .to_string(),
                    );
                    "DIVERGENT".to_string()
                } else if recovery.rolled_back_batches > 0 || recovery.stale_batches > 0 {
                    overall_ok = false;
                    recommendations.push(
                        "Run 'rmcp-memex repair-writes --execute' to clear rolled-back/stale recovery ledgers"
                            .to_string(),
                    );
                    "RECOVERY_PENDING".to_string()
                } else {
                    recommendations.push(
                        "Run 'rmcp-memex repair-writes --execute' to clear completed recovery ledgers"
                            .to_string(),
                    );
                    "LEDGER_PENDING".to_string()
                };

                Some(CrossStoreHealth {
                    recovery_dir: recovery.recovery_dir,
                    status,
                    pending_batches: recovery.pending_batches,
                    divergent_batches: recovery.divergent_batches,
                    rolled_back_batches: recovery.rolled_back_batches,
                    stale_batches: recovery.stale_batches,
                    missing_bm25_documents: recovery.documents_missing_bm25,
                    missing_lance_documents: recovery.documents_missing_lance,
                })
            }
            Err(error) => {
                overall_ok = false;
                recommendations.push(format!("BM25 recovery inspection unavailable: {}", error));
                Some(CrossStoreHealth {
                    recovery_dir: storage.cross_store_recovery_dir().display().to_string(),
                    status: format!("ERROR: {error}"),
                    pending_batches: 0,
                    divergent_batches: 0,
                    rolled_back_batches: 0,
                    stale_batches: 0,
                    missing_bm25_documents: 0,
                    missing_lance_documents: 0,
                })
            }
        },
        Err(_) => None,
    };

    // Build report
    let overall_status = if overall_ok && recommendations.is_empty() {
        "HEALTHY".to_string()
    } else if overall_ok {
        "OK (with recommendations)".to_string()
    } else {
        "UNHEALTHY".to_string()
    };

    let report = HealthReport {
        database: db_health,
        embedder: embedder_health,
        namespaces,
        cross_store,
        recommendations: recommendations.clone(),
        overall_status: overall_status.clone(),
    };

    // Output
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        // Config info
        if let Some(ref path) = config_path_display {
            eprintln!("Config: {}", path);
        }
        eprintln!();

        // Database section
        eprintln!("Database: {}", report.database.path);
        eprintln!("  Status:   {}", report.database.status);
        eprintln!("  Rows:     {}", report.database.row_count);
        eprintln!("  Versions: {}", report.database.version_count);
        eprintln!(
            "  Size:     ~{:.1} MB (estimate)",
            report.database.size_estimate_mb
        );
        eprintln!();

        // Embedder section
        if let Some(ref emb) = report.embedder {
            eprintln!("Embedder:");
            eprintln!("  Status:    {}", emb.status);
            if let Some(ref provider) = emb.provider {
                eprintln!("  Provider:  {}", provider);
            }
            if let Some(dim) = emb.dimension {
                let check = if emb.dimension_match {
                    "[OK]"
                } else {
                    "[MISMATCH]"
                };
                eprintln!("  Dimension: {} {}", dim, check);
            }
            eprintln!();
        } else if quick {
            eprintln!("Embedder: (skipped with --quick)");
            eprintln!();
        }

        // Namespaces section
        if !report.namespaces.is_empty() {
            eprintln!("Namespaces:");
            for ns in &report.namespaces {
                eprintln!("  {}: {} chunks", ns.name, ns.chunk_count);
            }
            eprintln!();
        }

        if let Some(ref cross_store) = report.cross_store {
            eprintln!("Cross-store recovery:");
            eprintln!("  Status:           {}", cross_store.status);
            eprintln!("  Recovery dir:     {}", cross_store.recovery_dir);
            eprintln!("  Pending batches:  {}", cross_store.pending_batches);
            eprintln!("  Divergent batches: {}", cross_store.divergent_batches);
            eprintln!("  Rolled back:      {}", cross_store.rolled_back_batches);
            eprintln!("  Stale ledgers:    {}", cross_store.stale_batches);
            eprintln!(
                "  Missing BM25 docs: {}",
                cross_store.missing_bm25_documents
            );
            eprintln!(
                "  Missing Lance docs: {}",
                cross_store.missing_lance_documents
            );
            eprintln!();
        }

        // Recommendations
        if report.recommendations.is_empty() {
            eprintln!("Status: {} - No action needed", overall_status);
        } else {
            eprintln!("Recommendations:");
            for rec in &report.recommendations {
                eprintln!("  - {}", rec);
            }
            eprintln!();
            eprintln!("Status: {}", overall_status);
        }
    }

    Ok(())
}

/// Recall result for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct RecallReport {
    pub query: String,
    pub summary: String,
    pub sources: Vec<RecallSource>,
    pub related: Vec<RecallRelated>,
    pub total_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallSource {
    pub namespace: String,
    pub source: Option<String>,
    pub date: Option<String>,
    pub preview: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallRelated {
    pub title: String,
    pub date: Option<String>,
    pub namespace: String,
}

/// Run recall command - synthesized search results
pub async fn run_recall(
    query: String,
    namespace_filter: Option<String>,
    limit: usize,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    use std::collections::HashMap;

    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    storage.ensure_collection().await?;

    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let rag = RAGPipeline::new(embedding_client, storage.clone()).await?;

    // Get namespaces to search
    let namespaces: Vec<String> = if let Some(ref ns) = namespace_filter {
        vec![ns.clone()]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    };

    if namespaces.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::json!({
                    "error": "No namespaces found",
                    "query": query,
                })
            );
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    // Search each namespace for outer layer results (summaries)
    let mut all_results: Vec<(String, rmcp_memex::SearchResult)> = Vec::new();

    for ns in &namespaces {
        // Search specifically for outer layer (summaries)
        let results = rag
            .memory_search_with_layer(ns, &query, limit, Some(SliceLayer::Outer))
            .await?;

        for r in results {
            all_results.push((ns.clone(), r));
        }
    }

    // Sort by score (best first)
    all_results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top results
    let top_results: Vec<_> = all_results.into_iter().take(limit).collect();

    if top_results.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::json!({
                    "query": query,
                    "summary": "No relevant memories found",
                    "sources": [],
                    "related": [],
                    "total_chunks": 0,
                })
            );
        } else {
            eprintln!("No relevant memories found for: \"{}\"", query);
        }
        return Ok(());
    }

    // Build summary from top outer slices
    let mut summary_parts: Vec<String> = Vec::new();
    let mut sources: Vec<RecallSource> = Vec::new();
    let mut seen_sources: HashMap<String, bool> = HashMap::new();

    for (ns, result) in &top_results {
        // Extract source file
        let source = result
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .or_else(|| result.metadata.get("file_path").and_then(|v| v.as_str()))
            .map(|s| {
                std::path::Path::new(s)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(s)
                    .to_string()
            });

        let date = result
            .metadata
            .get("indexed_at")
            .and_then(|v| v.as_str())
            .or_else(|| result.metadata.get("timestamp").and_then(|v| v.as_str()))
            .map(|s| s.split('T').next().unwrap_or(s).to_string());

        // Add to summary (first few, avoiding duplicates)
        let source_key = format!("{}-{:?}", ns, source);
        if !seen_sources.contains_key(&source_key) && summary_parts.len() < 5 {
            // Use the text directly as it's already a summary (outer layer)
            let text = result.text.trim();
            if !text.is_empty() && text.len() > 20 {
                summary_parts.push(text.to_string());
            }
            seen_sources.insert(source_key, true);
        }

        // Build preview
        let preview: String = result.text.chars().take(150).collect();
        let preview = if result.text.len() > 150 {
            format!("{}...", preview.trim())
        } else {
            preview.trim().to_string()
        };

        sources.push(RecallSource {
            namespace: ns.clone(),
            source,
            date,
            preview,
            score: result.score,
        });
    }

    // Build related items (unique sources/dates)
    let mut related: Vec<RecallRelated> = Vec::new();
    let mut seen_related: HashMap<String, bool> = HashMap::new();

    for source in &sources {
        let key = format!("{:?}-{:?}", source.source, source.date);
        if let std::collections::hash_map::Entry::Vacant(e) = seen_related.entry(key) {
            related.push(RecallRelated {
                title: source
                    .source
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
                date: source.date.clone(),
                namespace: source.namespace.clone(),
            });
            e.insert(true);
        }
    }

    // Combine summary parts
    let summary = if summary_parts.is_empty() {
        "Found relevant memories but no clear summary available.".to_string()
    } else {
        summary_parts.join("\n\n")
    };

    let report = RecallReport {
        query: query.clone(),
        summary,
        sources,
        related,
        total_chunks: top_results.len(),
    };

    // Output
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        eprintln!("Recall: \"{}\"", query);
        eprintln!();

        eprintln!("Based on {} relevant memories:", report.total_chunks);
        eprintln!();

        // Print summary
        for part in report.summary.lines().take(20) {
            eprintln!("  {}", part);
        }
        eprintln!();

        // Print related discussions
        if !report.related.is_empty() {
            eprintln!("Related discussions:");
            for r in report.related.iter().take(5) {
                let date_str = r.date.as_deref().unwrap_or("unknown date");
                eprintln!("  - \"{}\" ({}) [{}]", r.title, date_str, r.namespace);
            }
            if report.related.len() > 5 {
                eprintln!("  ... and {} more", report.related.len() - 5);
            }
            eprintln!();
        }

        // Namespace breakdown
        let mut ns_counts: HashMap<String, usize> = HashMap::new();
        for s in &report.sources {
            *ns_counts.entry(s.namespace.clone()).or_default() += 1;
        }
        let ns_summary: Vec<_> = ns_counts
            .iter()
            .map(|(k, v)| format!("{} from {}", v, k))
            .collect();
        eprintln!("Sources: {}", ns_summary.join(", "));
    }

    Ok(())
}

/// Timeline entry for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct TimelineEntry {
    pub date: String,
    pub namespace: String,
    pub source: Option<String>,
    pub chunk_count: usize,
}

/// Timeline report for JSON output
#[derive(Debug, Clone, Serialize)]
pub struct TimelineReport {
    pub namespaces: Vec<String>,
    pub entries: Vec<TimelineEntry>,
    pub coverage: TimelineCoverage,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineCoverage {
    pub earliest: Option<String>,
    pub latest: Option<String>,
    pub total_days: usize,
    pub days_with_data: usize,
}

/// Run timeline command - show when content was indexed
pub async fn run_timeline(
    db_path: String,
    namespace_filter: Option<String>,
    since: Option<String>,
    show_gaps_only: bool,
    json_output: bool,
) -> Result<()> {
    use std::collections::{BTreeMap, BTreeSet};

    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Get namespaces to query
    let namespaces: Vec<String> = if let Some(ref ns) = namespace_filter {
        vec![ns.clone()]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    };

    if namespaces.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::json!({
                    "error": "No namespaces found",
                    "namespaces": [],
                    "entries": [],
                })
            );
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    // Parse since filter
    let since_date: Option<chrono::NaiveDate> = since.as_ref().and_then(|s| {
        // Try parsing as duration like "30d"
        if let Some(days_str) = s.strip_suffix('d')
            && let Ok(days) = days_str.parse::<i64>()
        {
            return Some((chrono::Utc::now() - chrono::Duration::days(days)).date_naive());
        }
        // Try parsing as YYYY-MM
        if s.len() == 7
            && s.chars().nth(4) == Some('-')
            && let Ok(date) = chrono::NaiveDate::parse_from_str(&format!("{}-01", s), "%Y-%m-%d")
        {
            return Some(date);
        }
        // Try parsing as full date
        chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
    });

    // Collect timeline data: date -> namespace -> source -> count
    let mut timeline: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>> = BTreeMap::new();
    let mut all_dates: BTreeSet<String> = BTreeSet::new();

    for ns_name in &namespaces {
        let docs = storage.get_all_in_namespace(ns_name).await?;

        for doc in docs {
            // Extract indexed_at from metadata
            let indexed_at = doc
                .metadata
                .get("indexed_at")
                .and_then(|v| v.as_str())
                .or_else(|| doc.metadata.get("timestamp").and_then(|v| v.as_str()));

            let date_str = if let Some(ts) = indexed_at {
                // Parse ISO timestamp and extract date
                ts.split('T').next().unwrap_or("unknown").to_string()
            } else {
                "unknown".to_string()
            };

            // Apply since filter
            if let Some(since_d) = since_date
                && let Ok(doc_date) = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                && doc_date < since_d
            {
                continue;
            }

            all_dates.insert(date_str.clone());

            // Extract source from metadata
            let source = doc
                .metadata
                .get("source")
                .and_then(|v| v.as_str())
                .or_else(|| doc.metadata.get("file_path").and_then(|v| v.as_str()))
                .map(|s| {
                    // Extract filename from path
                    std::path::Path::new(s)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(s)
                        .to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());

            *timeline
                .entry(date_str)
                .or_default()
                .entry(ns_name.clone())
                .or_default()
                .entry(source)
                .or_default() += 1;
        }
    }

    // Build entries list for JSON
    let mut entries: Vec<TimelineEntry> = Vec::new();
    for (date, ns_map) in &timeline {
        for (ns, source_map) in ns_map {
            for (source, count) in source_map {
                entries.push(TimelineEntry {
                    date: date.clone(),
                    namespace: ns.clone(),
                    source: Some(source.clone()),
                    chunk_count: *count,
                });
            }
        }
    }

    // Calculate coverage
    let dates_vec: Vec<&String> = all_dates.iter().collect();
    let earliest = dates_vec.first().map(|s| (*s).clone());
    let latest = dates_vec.last().map(|s| (*s).clone());

    // Find gaps (consecutive dates with no data)
    let mut gaps: Vec<String> = Vec::new();
    if dates_vec.len() >= 2 {
        let sorted_dates: Vec<chrono::NaiveDate> = dates_vec
            .iter()
            .filter_map(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .collect();

        for window in sorted_dates.windows(2) {
            let diff = (window[1] - window[0]).num_days();
            if diff > 1 {
                gaps.push(format!(
                    "{} to {} ({} days)",
                    window[0].format("%Y-%m-%d"),
                    window[1].format("%Y-%m-%d"),
                    diff - 1
                ));
            }
        }
    }

    let coverage = TimelineCoverage {
        earliest,
        latest,
        total_days: if dates_vec.len() >= 2 {
            dates_vec
                .first()
                .and_then(|e| chrono::NaiveDate::parse_from_str(e, "%Y-%m-%d").ok())
                .zip(
                    dates_vec
                        .last()
                        .and_then(|l| chrono::NaiveDate::parse_from_str(l, "%Y-%m-%d").ok()),
                )
                .map(|(e, l)| (l - e).num_days() as usize + 1)
                .unwrap_or(0)
        } else {
            dates_vec.len()
        },
        days_with_data: all_dates.len(),
    };

    let report = TimelineReport {
        namespaces: namespaces.clone(),
        entries,
        coverage,
        gaps: gaps.clone(),
    };

    // Output
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if show_gaps_only {
        // Only show gaps
        if gaps.is_empty() {
            eprintln!("No gaps found in timeline");
        } else {
            eprintln!("Timeline Gaps:");
            for gap in &gaps {
                eprintln!("  - {}", gap);
            }
        }
    } else {
        // Full timeline grouped by month
        eprintln!("Timeline: {} namespace(s)", namespaces.len());
        if let Some(ref ns) = namespace_filter {
            eprintln!("  Namespace: {}", ns);
        }
        eprintln!();

        // Group by year-month
        let mut by_month: BTreeMap<String, Vec<(String, String, usize)>> = BTreeMap::new();
        for (date, ns_map) in &timeline {
            let month = if date.len() >= 7 {
                date[..7].to_string()
            } else {
                date.clone()
            };

            for (ns, source_map) in ns_map {
                let total: usize = source_map.values().sum();
                let sources: Vec<_> = source_map.keys().take(3).cloned().collect();
                let source_str = if sources.len() < source_map.len() {
                    format!(
                        "{} (+{} more)",
                        sources.join(", "),
                        source_map.len() - sources.len()
                    )
                } else {
                    sources.join(", ")
                };
                by_month.entry(month.clone()).or_default().push((
                    date.clone(),
                    format!("[{}] {} ({})", ns, source_str, total),
                    total,
                ));
            }
        }

        for (month, entries) in by_month {
            eprintln!("{}:", month);
            for (date, desc, _) in entries.iter().take(10) {
                eprintln!("  {}: {}", date, desc);
            }
            if entries.len() > 10 {
                eprintln!("  ... and {} more entries", entries.len() - 10);
            }
            eprintln!();
        }

        // Coverage summary
        eprintln!("Coverage:");
        if let (Some(e), Some(l)) = (&report.coverage.earliest, &report.coverage.latest) {
            eprintln!("  Period: {} to {}", e, l);
        }
        eprintln!(
            "  Days with data: {} / {} total",
            report.coverage.days_with_data, report.coverage.total_days
        );

        if !gaps.is_empty() {
            eprintln!();
            eprintln!("Gaps ({}):", gaps.len());
            for gap in gaps.iter().take(5) {
                eprintln!("  - {}", gap);
            }
            if gaps.len() > 5 {
                eprintln!("  ... and {} more gaps", gaps.len() - 5);
            }
        }
    }

    Ok(())
}

/// Run dive command - deep exploration with all onion layers
pub async fn run_dive(
    namespace: String,
    query: String,
    limit: usize,
    verbose: bool,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    // Search each layer separately
    let layers = [
        (Some(SliceLayer::Outer), "OUTER"),
        (Some(SliceLayer::Middle), "MIDDLE"),
        (Some(SliceLayer::Inner), "INNER"),
        (Some(SliceLayer::Core), "CORE"),
    ];

    let mut all_results: Vec<serde_json::Value> = Vec::new();

    if !json_output {
        eprintln!("\n=== DEEP DIVE: \"{}\" in [{}] ===\n", query, namespace);
    }

    for (layer_filter, layer_name) in &layers {
        let results = rag
            .memory_search_with_layer(&namespace, &query, limit, *layer_filter)
            .await?;

        if json_output {
            let layer_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let mut obj = serde_json::json!({
                        "id": r.id,
                        "score": r.score,
                        "keywords": r.keywords,
                        "layer": r.layer.map(|l| l.name()),
                        "can_expand": r.can_expand(),
                        "parent_id": r.parent_id,
                    });
                    if verbose {
                        obj["text"] = serde_json::json!(r.text);
                        obj["metadata"] = r.metadata.clone();
                        obj["children_ids"] = serde_json::json!(r.children_ids);
                    } else {
                        // Truncated preview
                        let preview: String = r.text.chars().take(200).collect();
                        obj["preview"] = serde_json::json!(preview);
                    }
                    obj
                })
                .collect();

            all_results.push(serde_json::json!({
                "layer": layer_name,
                "count": results.len(),
                "results": layer_results
            }));
        } else {
            eprintln!("--- {} LAYER ({} results) ---", layer_name, results.len());

            if results.is_empty() {
                eprintln!("  (no results)\n");
                continue;
            }

            for (i, result) in results.iter().enumerate() {
                eprintln!("  {}. [score: {:.3}] {}", i + 1, result.score, result.id);

                // Keywords
                if !result.keywords.is_empty() {
                    eprintln!("     Keywords: {}", result.keywords.join(", "));
                }

                // Text preview or full text
                if verbose {
                    eprintln!("     ---");
                    // Indent each line of text
                    for line in result.text.lines().take(20) {
                        eprintln!("     {}", line);
                    }
                    if result.text.lines().count() > 20 {
                        eprintln!("     ... ({} more lines)", result.text.lines().count() - 20);
                    }
                    eprintln!("     ---");

                    // Full metadata
                    if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
                        eprintln!("     Metadata: {}", result.metadata);
                    }
                } else {
                    // Short preview
                    let preview: String = result
                        .text
                        .chars()
                        .take(100)
                        .collect::<String>()
                        .replace('\n', " ");
                    let ellipsis = if result.text.len() > 100 { "..." } else { "" };
                    eprintln!("     \"{}{}\"", preview, ellipsis);
                }

                // Hierarchy info
                if result.can_expand() {
                    eprintln!("     [expandable: {} children]", result.children_ids.len());
                }
                if result.parent_id.is_some() {
                    eprintln!("     [has parent: can drill up]");
                }

                eprintln!();
            }
        }
    }

    if json_output {
        let output = serde_json::json!({
            "query": query,
            "namespace": namespace,
            "limit_per_layer": limit,
            "verbose": verbose,
            "layers": all_results
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!("=== END DIVE ===\n");
        eprintln!(
            "Tip: Use 'rmcp-memex expand -n {} -i <id>' to expand a result",
            namespace
        );
    }

    Ok(())
}

/// Run garbage collection
pub async fn run_gc(
    config: rmcp_memex::GcConfig,
    db_path: String,
    json_output: bool,
) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;

    let mode_str = if config.dry_run { "DRY RUN" } else { "EXECUTE" };
    let ns_str = config.namespace.as_deref().unwrap_or("all namespaces");

    if !json_output {
        eprintln!("\n=== GARBAGE COLLECTION ({}) ===\n", mode_str);
        eprintln!("Database: {}", db_path);
        eprintln!("Scope: {}", ns_str);
        eprintln!();

        if config.remove_orphans {
            eprintln!("- Checking for orphan embeddings...");
        }
        if config.remove_empty {
            eprintln!("- Checking for empty namespaces...");
        }
        if let Some(ref dur) = config.older_than {
            let days = dur.num_days();
            eprintln!("- Checking for documents older than {} days...", days);
        }
        eprintln!();
    }

    // Run GC
    let stats = storage.garbage_collect(&config).await?;

    if json_output {
        let output = serde_json::json!({
            "mode": if config.dry_run { "dry_run" } else { "execute" },
            "db_path": db_path,
            "namespace": config.namespace,
            "orphans": {
                "found": stats.orphans_found,
                "removed": stats.orphans_removed
            },
            "empty_namespaces": {
                "found": stats.empty_namespaces_found,
                "removed": stats.empty_namespaces_removed,
                "names": stats.empty_namespace_names
            },
            "old_documents": {
                "found": stats.old_docs_found,
                "removed": stats.old_docs_removed,
                "affected_namespaces": stats.affected_namespaces
            },
            "bytes_freed": stats.bytes_freed,
            "has_issues": stats.has_issues(),
            "has_deletions": stats.has_deletions()
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Human-readable output
        eprintln!("=== RESULTS ===\n");

        // Orphans
        if config.remove_orphans {
            if stats.orphans_found > 0 {
                eprintln!("Orphan embeddings:");
                eprintln!("  Found:   {}", stats.orphans_found);
                if config.dry_run {
                    eprintln!("  Action:  Would remove {} orphans", stats.orphans_found);
                } else {
                    eprintln!("  Removed: {}", stats.orphans_removed);
                }
            } else {
                eprintln!("Orphan embeddings: None found");
            }
            eprintln!();
        }

        // Empty namespaces
        if config.remove_empty {
            if stats.empty_namespaces_found > 0 {
                eprintln!("Empty namespaces:");
                eprintln!("  Found: {}", stats.empty_namespaces_found);
                for ns in &stats.empty_namespace_names {
                    eprintln!("    - {}", ns);
                }
            } else {
                eprintln!("Empty namespaces: None found");
            }
            eprintln!();
        }

        // Old documents
        if config.older_than.is_some() {
            if stats.old_docs_found > 0 {
                eprintln!("Old documents:");
                eprintln!("  Found:   {}", stats.old_docs_found);
                if config.dry_run {
                    eprintln!("  Action:  Would remove {} documents", stats.old_docs_found);
                } else {
                    eprintln!("  Removed: {}", stats.old_docs_removed);
                }
                if !stats.affected_namespaces.is_empty() {
                    eprintln!("  Affected namespaces:");
                    for ns in &stats.affected_namespaces {
                        eprintln!("    - {}", ns);
                    }
                }
            } else {
                eprintln!("Old documents: None found (no documents with parseable timestamps)");
            }
            eprintln!();
        }

        // Summary
        eprintln!("=== SUMMARY ===\n");
        if !stats.has_issues() {
            eprintln!("No issues found. Database is clean.");
        } else if config.dry_run {
            eprintln!("Issues found. Run with --execute to apply changes.");
            eprintln!();
            eprintln!("Example:");
            let mut cmd = "rmcp-memex gc".to_string();
            if config.remove_orphans {
                cmd.push_str(" --remove-orphans");
            }
            if config.remove_empty {
                cmd.push_str(" --remove-empty");
            }
            if let Some(ref dur) = config.older_than {
                cmd.push_str(&format!(" --older-than {}d", dur.num_days()));
            }
            cmd.push_str(" --execute");
            eprintln!("  {}", cmd);
        } else if stats.has_deletions() {
            eprintln!("Cleanup complete!");
            let total_removed = stats.orphans_removed + stats.old_docs_removed;
            eprintln!("  Total items removed: {}", total_removed);
            if let Some(bytes) = stats.bytes_freed {
                eprintln!("  Space freed: {} bytes", bytes);
            }
            eprintln!();
            eprintln!("Tip: Run 'rmcp-memex optimize' to compact the database and reclaim space.");
        }
    }

    Ok(())
}
