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

/// Validate that an auth token contains only ASCII bytes (RFC 7230).
/// Returns Ok(()) if valid, Err with descriptive message if non-ASCII byte found.
fn validate_ascii_token(token: &str) -> Result<()> {
    for (pos, byte) in token.bytes().enumerate() {
        if !byte.is_ascii() {
            return Err(anyhow::anyhow!(
                "ERROR: --auth-token must be ASCII (RFC 7230). Got non-ASCII byte 0x{:02x} at position {}.",
                byte,
                pos
            ));
        }
    }
    Ok(())
}

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

    let auth_mode_str = file_cfg.auth_mode.as_deref().unwrap_or("mutating-only");
    // CLI flag overrides file config
    let auth_mode = rmcp_memex::http::AuthMode::parse(if cli.auth_mode != "mutating-only" {
        &cli.auth_mode
    } else {
        auth_mode_str
    });
    let allow_query_token = cli.allow_query_token || file_cfg.allow_query_token.unwrap_or(false);

    rmcp_memex::http::HttpServerConfig {
        auth_token,
        cors_origins,
        bind_address,
        auth_mode,
        allow_query_token,
        auth_manager: None, // initialized lazily in start_server if NamespaceAcl mode
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

/// Validate startup preconditions for the HTTP server:
/// - ASCII guard on auth token
/// - Non-loopback bind without auth is a hard error (unless escape hatch)
fn validate_http_preconditions(
    http_config: &rmcp_memex::http::HttpServerConfig,
    allow_network_without_auth: bool,
) -> Result<()> {
    // ASCII guard
    if let Some(ref token) = http_config.auth_token {
        validate_ascii_token(token)?;
    }

    // Bind guard: non-loopback without auth
    if !http_config.bind_address.is_loopback() && http_config.auth_token.is_none() {
        if allow_network_without_auth {
            eprintln!(
                "WARNING: HTTP server exposed on network without auth token. \
                 This is allowed via --allow-network-without-auth but is NOT recommended."
            );
        } else {
            return Err(anyhow::anyhow!(
                "ERROR: Refusing to bind to {} without --auth-token. \
                 Network-exposed server without authentication is a security risk.\n\
                 Options:\n  \
                 1. Add --auth-token <token> or set MEMEX_AUTH_TOKEN\n  \
                 2. Add --allow-network-without-auth to override (not recommended)",
                http_config.bind_address
            ));
        }
    }

    Ok(())
}

async fn run_http_only_command(cli: Cli, port: u16, auto_open_browser: bool) -> Result<()> {
    let (file_cfg, _) = load_or_discover_config(cli.config.as_deref())?;
    let http_server_config = resolve_http_server_config(&cli, &file_cfg);
    validate_http_preconditions(&http_server_config, cli.allow_network_without_auth)?;
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
        Some(Commands::Auth { action }) => run_auth_command(action, cli.token_store_path).await,
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
            // Validate HTTP preconditions (ASCII guard, bind guard) before starting
            if http_port.is_some() || http_only {
                validate_http_preconditions(&http_server_config, cli.allow_network_without_auth)?;
            }
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

async fn run_auth_command(action: AuthAction, token_store_path: Option<String>) -> Result<()> {
    use rmcp_memex::auth::{Scope, TokenStoreFile};
    use std::str::FromStr;

    let store_path =
        token_store_path.unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/tokens.json".to_string());
    let store = TokenStoreFile::new(store_path.clone());
    store.load().await?;

    match action {
        AuthAction::Create {
            id,
            description,
            scopes,
            namespaces,
            expires_at,
            json,
        } => {
            let token_id = id.unwrap_or_else(|| {
                use uuid::Uuid;
                Uuid::new_v4().to_string()[..8].to_string()
            });

            let parsed_scopes: Vec<Scope> = scopes
                .split(',')
                .map(|s| Scope::from_str(s.trim()))
                .collect::<Result<Vec<_>>>()?;

            let parsed_namespaces: Vec<String> = namespaces
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            let parsed_expiry = if let Some(exp_str) = expires_at {
                Some(
                    chrono::DateTime::parse_from_rfc3339(&exp_str)
                        .map_err(|e| anyhow::anyhow!("Invalid expiry format: {}. Use RFC 3339 (e.g., 2026-12-31T00:00:00Z)", e))?
                        .with_timezone(&chrono::Utc),
                )
            } else {
                None
            };

            let plaintext = store
                .create_token(
                    token_id.clone(),
                    parsed_scopes.clone(),
                    parsed_namespaces.clone(),
                    parsed_expiry,
                    description.clone(),
                )
                .await?;

            if json {
                let output = serde_json::json!({
                    "id": token_id,
                    "token": plaintext,
                    "scopes": parsed_scopes,
                    "namespaces": parsed_namespaces,
                    "description": description,
                    "store_path": store_path,
                    "warning": "This token is shown ONCE. Store it securely."
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!("Token created successfully.");
                eprintln!();
                eprintln!("  ID:          {}", token_id);
                eprintln!("  Scopes:      {}", scopes);
                eprintln!("  Namespaces:  {}", namespaces);
                eprintln!("  Description: {}", description);
                eprintln!("  Store:       {}", store_path);
                eprintln!();
                eprintln!("  TOKEN (shown ONCE, store securely):");
                println!("{}", plaintext);
                eprintln!();
                eprintln!("  Use as: Authorization: Bearer {}", plaintext);
            }

            Ok(())
        }
        AuthAction::List { json } => {
            let tokens = store.list_tokens().await;

            if json {
                let output: Vec<serde_json::Value> = tokens
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "id": t.id,
                            "scopes": t.scopes,
                            "namespaces": t.namespaces,
                            "expires_at": t.expires_at,
                            "description": t.description,
                            "created_at": t.created_at,
                            "expired": t.is_expired(),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else if tokens.is_empty() {
                eprintln!("No tokens configured.");
                eprintln!("Create one with: rmcp-memex auth create --description \"My token\"");
            } else {
                eprintln!("{} token(s) in store:", tokens.len());
                eprintln!();
                for t in &tokens {
                    let scopes_str: Vec<String> = t.scopes.iter().map(|s| s.to_string()).collect();
                    let expired_marker = if t.is_expired() { " [EXPIRED]" } else { "" };
                    eprintln!("  {} {}", t.id, expired_marker);
                    eprintln!("    Description: {}", t.description);
                    eprintln!("    Scopes:      [{}]", scopes_str.join(", "));
                    eprintln!("    Namespaces:  [{}]", t.namespaces.join(", "));
                    if let Some(exp) = t.expires_at {
                        eprintln!("    Expires:     {}", exp.to_rfc3339());
                    } else {
                        eprintln!("    Expires:     never");
                    }
                    eprintln!("    Created:     {}", t.created_at.to_rfc3339());
                    eprintln!();
                }
            }

            Ok(())
        }
        AuthAction::Revoke { id } => {
            let removed = store.revoke_token(&id).await?;
            if removed {
                eprintln!("Token '{}' revoked.", id);
            } else {
                eprintln!("Token '{}' not found.", id);
            }
            Ok(())
        }
        AuthAction::Rotate { id, json } => {
            let new_plaintext = store.rotate_token(&id).await?;

            if json {
                let output = serde_json::json!({
                    "id": id,
                    "token": new_plaintext,
                    "warning": "This token is shown ONCE. Store it securely."
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!("Token '{}' rotated.", id);
                eprintln!();
                eprintln!("  NEW TOKEN (shown ONCE, store securely):");
                println!("{}", new_plaintext);
                eprintln!();
                eprintln!("  Use as: Authorization: Bearer {}", new_plaintext);
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn ascii_guard_accepts_ascii_token() {
        assert!(validate_ascii_token("my-secure-token-123").is_ok());
        assert!(validate_ascii_token("abcABC012!@#$%").is_ok());
        assert!(validate_ascii_token("").is_ok()); // empty is technically ASCII
    }

    #[test]
    fn ascii_guard_rejects_non_ascii_token() {
        let err = validate_ascii_token("token-with-\u{015b}-polish").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ASCII"),
            "Error message should mention ASCII: {msg}"
        );
        assert!(
            msg.contains("0xc5"),
            "Error message should show the offending byte: {msg}"
        );
    }

    #[test]
    fn bind_guard_blocks_network_without_auth() {
        let config = rmcp_memex::http::HttpServerConfig {
            bind_address: Ipv4Addr::UNSPECIFIED.into(),
            auth_token: None,
            ..Default::default()
        };
        let result = validate_http_preconditions(&config, false);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Refusing to bind"),
            "Should contain refusal: {msg}"
        );
    }

    #[test]
    fn bind_guard_allows_network_with_escape_hatch() {
        let config = rmcp_memex::http::HttpServerConfig {
            bind_address: Ipv4Addr::UNSPECIFIED.into(),
            auth_token: None,
            ..Default::default()
        };
        assert!(validate_http_preconditions(&config, true).is_ok());
    }

    #[test]
    fn bind_guard_allows_network_with_auth() {
        let config = rmcp_memex::http::HttpServerConfig {
            bind_address: Ipv4Addr::UNSPECIFIED.into(),
            auth_token: Some("my-token".to_string()),
            ..Default::default()
        };
        assert!(validate_http_preconditions(&config, false).is_ok());
    }

    #[test]
    fn bind_guard_allows_localhost_without_auth() {
        let config = rmcp_memex::http::HttpServerConfig {
            bind_address: Ipv4Addr::LOCALHOST.into(),
            auth_token: None,
            ..Default::default()
        };
        assert!(validate_http_preconditions(&config, false).is_ok());
    }

    #[test]
    fn ascii_guard_rejects_non_ascii_in_preconditions() {
        let config = rmcp_memex::http::HttpServerConfig {
            auth_token: Some("token-\u{0107}".to_string()), // \u{0107} = 'c' with acute
            ..Default::default()
        };
        let result = validate_http_preconditions(&config, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ASCII"));
    }
}
