use anyhow::Result;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::process::Command as ProcessCommand;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::info;
use tracing_subscriber::FmtSubscriber;

use rmcp_memex::{
    EmbeddingClient, QueryRouter, RAGPipeline, SearchMode, SearchModeRecommendation, SliceLayer,
    SliceMode, StorageManager, WizardConfig, create_server, run_wizard,
};

use crate::cli::config::*;
use crate::cli::data::*;
use crate::cli::definition::*;
use crate::cli::inspection::*;
use crate::cli::maintenance::*;
use crate::cli::search::*;

fn resolve_http_server_config(
    cli: &Cli,
    file_cfg: &FileConfig,
) -> rmcp_memex::http::HttpServerConfig {
    let auth_token = cli
        .auth_token
        .clone()
        .or_else(|| std::env::var("MEMEX_AUTH_TOKEN").ok())
        .or_else(|| file_cfg.auth_token.clone());
    let bind_addr_str = cli
        .bind_address
        .clone()
        .or_else(|| file_cfg.bind_address.clone())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let bind_address: IpAddr = bind_addr_str.parse().unwrap_or_else(|_| {
        eprintln!(
            "Invalid bind address '{}', falling back to 127.0.0.1",
            bind_addr_str
        );
        Ipv4Addr::LOCALHOST.into()
    });
    let cors_origins: Vec<String> = cli
        .cors_origins
        .clone()
        .or_else(|| file_cfg.cors_origins.clone())
        .map(|s| {
            s.split(',')
                .map(|o| o.trim().to_string())
                .filter(|o| !o.is_empty())
                .collect()
        })
        .unwrap_or_default();

    rmcp_memex::http::HttpServerConfig {
        auth_token,
        cors_origins,
        bind_address,
    }
}

fn dashboard_browser_url(bind_address: IpAddr, port: u16) -> String {
    let host = match bind_address {
        IpAddr::V4(addr) if addr.is_unspecified() => Ipv4Addr::LOCALHOST.to_string(),
        IpAddr::V4(addr) => addr.to_string(),
        IpAddr::V6(addr) if addr.is_unspecified() => format!("[{}]", Ipv6Addr::LOCALHOST),
        IpAddr::V6(addr) => format!("[{addr}]"),
    };

    format!("http://{host}:{port}/")
}

fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        ProcessCommand::new("open").arg(url).spawn()?;
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        ProcessCommand::new("xdg-open").arg(url).spawn()?;
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        ProcessCommand::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err(anyhow::anyhow!(
        "Automatic browser open is not supported on this platform"
    ))
}

async fn run_http_only_command(cli: Cli, port: u16, auto_open_browser: bool) -> Result<()> {
    let (file_cfg, _) = load_or_discover_config(cli.config.as_deref())?;
    let http_server_config = resolve_http_server_config(&cli, &file_cfg);
    let dashboard_url = dashboard_browser_url(http_server_config.bind_address, port);

    let mut config = cli.into_server_config()?;
    config.hybrid.bm25.read_only = true;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(config.log_level)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting RMCP Memex");
    info!("Cache: {}MB", config.cache_mb);
    info!("DB Path: {}", config.db_path);

    let server = create_server(config).await?;
    let mcp_core = server.mcp_core();

    if auto_open_browser {
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(350)).await;
            if let Err(err) = open_browser(&dashboard_url) {
                eprintln!("Warning: failed to open dashboard browser: {}", err);
            }
        });
    }

    rmcp_memex::http::start_server(mcp_core, port, http_server_config).await
}

pub async fn run_command(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Commands::Dashboard { port, no_open }) => {
            let port = port.or(cli.http_port).unwrap_or(DEFAULT_DASHBOARD_PORT);
            run_http_only_command(cli, port, !no_open).await
        }
        Some(Commands::Sse { port }) => {
            let port = port.or(cli.http_port).unwrap_or(DEFAULT_SSE_PORT);
            run_http_only_command(cli, port, false).await
        }
        Some(Commands::Wizard { dry_run }) => {
            let wizard_config = WizardConfig {
                config_path: cli.config,
                dry_run,
            };
            run_wizard(wizard_config)
        }
        Some(Commands::Index {
            path,
            namespace,
            recursive,
            glob,
            max_depth,
            preprocess,
            sanitize_metadata,
            slice_mode,
            dedup,
            progress,
            resume,
            pipeline,
            pipeline_embed_concurrency,
            pipeline_governor,
            parallel,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            let _cache_mb = cli.cache_mb.or(cfg.file_cfg.cache_mb).unwrap_or(4096);
            let preprocess = preprocess || cfg.file_cfg.preprocessing_enabled.unwrap_or(false);
            let slice_mode: SliceMode = slice_mode.parse().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid slice mode '{}'. Use one of: flat, onion, onion-fast",
                    slice_mode
                )
            })?;

            let result = run_batch_index(BatchIndexConfig {
                path,
                namespace,
                recursive,
                glob_pattern: glob,
                max_depth,
                db_path: cfg.db_path.clone(),
                preprocess,
                sanitize_metadata,
                slice_mode,
                dedup,
                embedding_config: cfg.embedding_config,
                show_progress: progress,
                resume,
                pipeline,
                pipeline_embed_concurrency,
                pipeline_governor,
                parallel,
            })
            .await;

            if result.is_ok()
                && let Ok(storage) = StorageManager::new_lance_only(&cfg.db_path).await
            {
                let _ = check_and_maybe_optimize(&storage, &cfg.maintenance_config).await;
            }

            result
        }
        Some(Commands::Overview { namespace, json }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_overview(namespace, json, cfg.db_path).await
        }
        Some(Commands::Dive {
            namespace,
            query,
            limit,
            verbose,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_dive(
                namespace,
                query,
                limit,
                verbose,
                json,
                cfg.db_path,
                &cfg.embedding_config,
            )
            .await
        }
        Some(Commands::Search {
            namespace,
            query,
            limit,
            json,
            deep,
            layer,
            mode,
            auto_route,
            ..
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;

            let layer_filter = if deep {
                None
            } else if let Some(layer_str) = layer {
                match layer_str.as_str() {
                    "outer" => Some(SliceLayer::Outer),
                    "middle" => Some(SliceLayer::Middle),
                    "inner" => Some(SliceLayer::Inner),
                    "core" => Some(SliceLayer::Core),
                    _ => None,
                }
            } else {
                Some(SliceLayer::Outer)
            };

            let search_mode: SearchMode = if auto_route {
                let router = QueryRouter::new();
                let decision = router.route(&query);
                eprintln!(
                    "Query intent: {} (confidence: {:.2})",
                    decision.intent, decision.confidence
                );
                if let Some(ref suggestion) = decision.loctree_suggestion {
                    eprintln!(
                        "Consider: {} - {}",
                        suggestion.command, suggestion.explanation
                    );
                }
                if let Some(ref hints) = decision.temporal_hints
                    && !hints.date_references.is_empty()
                {
                    eprintln!("Date references: {}", hints.date_references.join(", "));
                }
                match decision.recommended_mode.mode {
                    SearchModeRecommendation::Vector => SearchMode::Vector,
                    SearchModeRecommendation::Bm25 => SearchMode::Keyword,
                    SearchModeRecommendation::Hybrid => SearchMode::Hybrid,
                }
            } else {
                mode.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "Invalid search mode '{}'. Use one of: vector, keyword, hybrid, auto",
                        mode
                    )
                })?
            };

            run_search(SearchConfig {
                namespace,
                query,
                limit,
                json_output: json,
                db_path: cfg.db_path,
                layer_filter,
                search_mode,
                embedding_config: &cfg.embedding_config,
            })
            .await
        }
        Some(Commands::Expand {
            namespace,
            id,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_expand(namespace, id, json, cfg.db_path, &cfg.embedding_config).await
        }
        Some(Commands::Get {
            namespace,
            id,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_get(namespace, id, json, cfg.db_path, &cfg.embedding_config).await
        }
        Some(Commands::RagSearch {
            query,
            limit,
            namespace,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_rag_search(
                query,
                limit,
                namespace,
                json,
                cfg.db_path,
                &cfg.embedding_config,
            )
            .await
        }
        Some(Commands::Namespaces { stats, json }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_list_namespaces(stats, json, cfg.db_path).await
        }
        Some(Commands::Export {
            namespace,
            output,
            include_embeddings,
            db_path: cmd_db_path,
        }) => {
            let file_cfg = load_or_discover_config(cli.config.as_deref())?.0;
            let db_path = cmd_db_path
                .or(cli.db_path)
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();
            run_export(namespace, output, include_embeddings, db_path).await
        }
        Some(Commands::Upsert {
            namespace,
            id,
            text,
            metadata,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            let content = match text {
                Some(t) => t,
                None => {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin().read_to_string(&mut buffer)?;
                    buffer
                }
            };
            if content.trim().is_empty() {
                return Err(anyhow::anyhow!("No text provided"));
            }
            let meta: serde_json::Value = serde_json::from_str(&metadata)
                .map_err(|e| anyhow::anyhow!("Invalid metadata JSON: {}", e))?;
            let embedding_client = Arc::new(Mutex::new(
                EmbeddingClient::new(&cfg.embedding_config).await?,
            ));
            let storage = Arc::new(StorageManager::new_lance_only(&cfg.db_path).await?);
            let rag = RAGPipeline::new(embedding_client, storage.clone()).await?;
            rag.memory_upsert(&namespace, id.clone(), content.clone(), meta)
                .await?;
            eprintln!("✓ Upserted chunk '{}' to namespace '{}'", id, namespace);
            eprintln!("  Text: {} chars", content.len());
            eprintln!("  DB: {}", cfg.db_path);
            let _ = check_and_maybe_optimize(&storage, &cfg.maintenance_config).await;
            Ok(())
        }
        Some(Commands::Optimize) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            eprintln!("Optimizing database at: {}", cfg.db_path);
            let storage = StorageManager::new_lance_only(&cfg.db_path).await?;
            let stats = storage.optimize().await?;
            eprintln!("\nOptimization complete:");
            if let Some(ref c) = stats.compaction {
                eprintln!("  Files rewritten:    {}", c.files_removed);
                eprintln!("  Files added:        {}", c.files_added);
                eprintln!("  Fragments removed:  {}", c.fragments_removed);
                eprintln!("  Fragments added:    {}", c.fragments_added);
            }
            if let Some(ref p) = stats.prune {
                eprintln!("  Versions removed:   {}", p.old_versions);
                eprintln!("  Bytes freed:        {}", p.bytes_removed);
            }
            Ok(())
        }
        Some(Commands::Health { quick, json }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_health(
                cfg.db_path,
                &cfg.embedding_config,
                cfg.config_path,
                quick,
                json,
            )
            .await
        }
        Some(Commands::Recall {
            query,
            namespace,
            limit,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_recall(
                query,
                namespace,
                limit,
                json,
                cfg.db_path,
                &cfg.embedding_config,
            )
            .await
        }
        Some(Commands::Timeline {
            namespace,
            since,
            gaps,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_timeline(cfg.db_path, namespace, since, gaps, json).await
        }
        Some(Commands::Compact) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            eprintln!("Compacting database at: {}", cfg.db_path);
            let storage = StorageManager::new_lance_only(&cfg.db_path).await?;
            let stats = storage.compact().await?;
            eprintln!("\nCompaction complete:");
            if let Some(ref c) = stats.compaction {
                eprintln!("  Files rewritten:    {}", c.files_removed);
                eprintln!("  Files added:        {}", c.files_added);
                eprintln!("  Fragments removed:  {}", c.fragments_removed);
                eprintln!("  Fragments added:    {}", c.fragments_added);
            } else {
                eprintln!("  No compaction needed");
            }
            Ok(())
        }
        Some(Commands::Cleanup { older_than_days }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            eprintln!(
                "Cleaning up versions older than {} days at: {}",
                older_than_days, cfg.db_path
            );
            let storage = StorageManager::new_lance_only(&cfg.db_path).await?;
            let stats = storage.cleanup(Some(older_than_days)).await?;
            eprintln!("\nCleanup complete:");
            if let Some(ref p) = stats.prune {
                eprintln!("  Versions removed:   {}", p.old_versions);
                eprintln!("  Bytes freed:        {}", p.bytes_removed);
            } else {
                eprintln!("  No old versions to remove");
            }
            Ok(())
        }
        Some(Commands::Stats) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            let storage = StorageManager::new_lance_only(&cfg.db_path).await?;
            let stats = storage.stats().await?;
            eprintln!("Database Statistics:");
            eprintln!("  Table:       {}", stats.table_name);
            eprintln!("  Path:        {}", stats.db_path);
            eprintln!("  Total rows:  {}", stats.row_count);
            eprintln!("  Versions:    {}", stats.version_count);
            println!("{}", serde_json::to_string_pretty(&stats)?);
            Ok(())
        }
        Some(Commands::Gc {
            remove_orphans,
            remove_empty,
            older_than,
            execute,
            namespace,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            if !remove_orphans && !remove_empty && older_than.is_none() {
                return Err(anyhow::anyhow!(
                    "No GC operation specified. Use --remove-orphans, --remove-empty, or --older-than <duration>"
                ));
            }
            let older_than_duration = if let Some(dur_str) = older_than {
                Some(rmcp_memex::parse_duration_string(&dur_str)?)
            } else {
                None
            };
            let gc_config = rmcp_memex::GcConfig {
                remove_orphans,
                remove_empty,
                older_than: older_than_duration,
                dry_run: !execute,
                namespace,
            };
            run_gc(gc_config, cfg.db_path, json).await
        }
        Some(Commands::RepairWrites {
            namespace,
            execute,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_repair_writes(cfg.db_path, namespace, execute, json).await
        }
        Some(Commands::CrossSearch {
            query,
            limit,
            total_limit,
            mode,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_cross_search(
                query,
                limit,
                total_limit,
                mode,
                json,
                cfg.db_path,
                &cfg.embedding_config,
            )
            .await
        }
        Some(Commands::Merge {
            source,
            target,
            dedup,
            namespace_prefix,
            dry_run,
            json,
        }) => run_merge(source, target, dedup, namespace_prefix, dry_run, json).await,
        Some(Commands::Dedup {
            namespace,
            dry_run,
            keep,
            cross_namespace,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_dedup(
                namespace,
                dry_run,
                KeepStrategy::from_str(&keep),
                cross_namespace,
                json,
                cfg.db_path,
            )
            .await
        }
        Some(Commands::MigrateNamespace {
            from,
            to,
            merge,
            delete_source,
            dry_run,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_migrate_namespace(from, to, cfg.db_path, merge, delete_source, dry_run, json).await
        }
        Some(Commands::PurgeNamespace {
            namespace,
            confirm,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_purge_namespace(namespace, cfg.db_path, confirm, json).await
        }
        Some(Commands::Import {
            namespace,
            input,
            skip_existing,
            db_path: cmd_db_path,
        }) => {
            let file_cfg = load_or_discover_config(cli.config.as_deref())?.0;
            let embedding_config = file_cfg.resolve_embedding_config();
            let db_path = cmd_db_path
                .or(cli.db_path)
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();
            run_import(namespace, input, skip_existing, db_path, &embedding_config).await
        }
        Some(Commands::Reprocess {
            namespace,
            input,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            db_path: cmd_db_path,
        }) => {
            let file_cfg = load_or_discover_config(cli.config.as_deref())?.0;
            let embedding_config = file_cfg.resolve_embedding_config();
            let db_path = cmd_db_path
                .or(cli.db_path)
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();
            let slice_mode: SliceMode = slice_mode.parse().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid slice mode '{}'. Use one of: flat, onion, onion-fast",
                    slice_mode
                )
            })?;
            run_reprocess(
                ReprocessConfig {
                    namespace,
                    input,
                    slice_mode,
                    preprocess,
                    skip_existing,
                    dry_run,
                    db_path,
                },
                &embedding_config,
            )
            .await
        }
        Some(Commands::Reindex {
            namespace,
            target_namespace,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            db_path: cmd_db_path,
        }) => {
            let file_cfg = load_or_discover_config(cli.config.as_deref())?.0;
            let embedding_config = file_cfg.resolve_embedding_config();
            let db_path = cmd_db_path
                .or(cli.db_path)
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();
            let slice_mode: SliceMode = slice_mode.parse().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid slice mode '{}'. Use one of: flat, onion, onion-fast",
                    slice_mode
                )
            })?;
            let target_namespace =
                target_namespace.unwrap_or_else(|| default_reindexed_namespace(&namespace));
            run_reindex(
                ReindexConfig {
                    source_namespace: namespace,
                    target_namespace,
                    slice_mode,
                    preprocess,
                    skip_existing,
                    dry_run,
                    db_path,
                },
                &embedding_config,
            )
            .await
        }
        Some(Commands::Audit {
            namespace,
            threshold,
            verbose,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_audit(namespace, threshold, verbose, json, cfg.db_path).await
        }
        Some(Commands::PurgeQuality {
            threshold,
            confirm,
            json,
        }) => {
            let cfg = ResolvedConfig::load(cli.config.as_deref(), cli.db_path.as_deref())?;
            run_purge_quality(threshold, confirm, json, cfg.db_path).await
        }
        Some(Commands::Serve) | None => {
            let http_port = cli.http_port;
            let http_only = cli.http_only;
            if http_only && http_port.is_none() {
                return Err(anyhow::anyhow!(
                    "--http-only requires --http-port to be set"
                ));
            }
            let (file_cfg_ref, _) = load_or_discover_config(cli.config.as_deref())?;
            let http_server_config = resolve_http_server_config(&cli, &file_cfg_ref);
            let mut config = cli.into_server_config()?;
            if http_only {
                config.hybrid.bm25.read_only = true;
            }
            let subscriber = FmtSubscriber::builder()
                .with_max_level(config.log_level)
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;
            info!("Starting RMCP Memex");
            info!("Cache: {}MB", config.cache_mb);
            info!("DB Path: {}", config.db_path);
            let server = create_server(config).await?;
            if http_only {
                let port = http_port.expect("validated above");
                let mcp_core = server.mcp_core();
                info!("Starting HTTP-only server on port {} (no MCP stdio)", port);
                rmcp_memex::http::start_server(mcp_core, port, http_server_config).await?;
                return Ok(());
            }
            if let Some(port) = http_port {
                let mcp_core = server.mcp_core();
                info!("Starting HTTP/SSE server on port {}", port);
                tokio::spawn(async move {
                    if let Err(e) =
                        rmcp_memex::http::start_server(mcp_core, port, http_server_config).await
                    {
                        tracing::error!("HTTP server error: {}", e);
                    }
                });
            }
            server.run_stdio().await
        }
    }
}
