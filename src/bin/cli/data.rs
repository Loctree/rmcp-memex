use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use rmcp_memex::{
    EmbeddingClient, EmbeddingConfig, PreprocessingConfig, Preprocessor, RAGPipeline, SliceMode,
    StorageManager, compute_content_hash, path_utils,
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
/// JSONL export record structure
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportRecord {
    pub id: String,
    pub text: String,
    pub metadata: Value,
    pub content_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct ReprocessDocument {
    canonical_id: String,
    source_record_id: String,
    text: String,
    metadata: Value,
    source_text_hash: String,
    collapsed_records: usize,
}

struct RebuildPlan {
    namespace: String,
    db_path: String,
    source_label: String,
    source_records: usize,
    docs: Vec<ReprocessDocument>,
    slice_mode: SliceMode,
    preprocess: bool,
    skip_existing: bool,
    dry_run: bool,
    parse_errors: usize,
}

pub struct ReprocessConfig {
    pub namespace: String,
    pub input: PathBuf,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
    pub db_path: String,
}

pub struct ReindexConfig {
    pub source_namespace: String,
    pub target_namespace: String,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
    pub db_path: String,
}

pub fn default_reindexed_namespace(namespace: &str) -> String {
    format!("{namespace}-reindexed")
}

fn preferred_reprocess_id(record: &ExportRecord) -> String {
    record
        .metadata
        .get("original_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            record
                .metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| record.id.clone())
}

fn reprocess_layer_rank(metadata: &Value) -> u8 {
    match metadata.get("layer").and_then(Value::as_str) {
        Some("core") => 4,
        Some("inner") => 3,
        Some("middle") => 2,
        Some("outer") => 1,
        _ => 0,
    }
}

fn should_replace_reprocess_candidate(
    current: &ReprocessDocument,
    candidate: &ReprocessDocument,
) -> bool {
    let current_rank = (current.text.len(), reprocess_layer_rank(&current.metadata));
    let candidate_rank = (
        candidate.text.len(),
        reprocess_layer_rank(&candidate.metadata),
    );
    candidate_rank > current_rank
}

fn collapse_export_records(records: Vec<ExportRecord>) -> Vec<ReprocessDocument> {
    let mut grouped: HashMap<String, ReprocessDocument> = HashMap::new();

    for record in records {
        let text = record.text.trim();
        if text.is_empty() {
            continue;
        }

        let candidate = ReprocessDocument {
            canonical_id: preferred_reprocess_id(&record),
            source_record_id: record.id,
            text: text.to_string(),
            metadata: record.metadata,
            source_text_hash: compute_content_hash(text),
            collapsed_records: 1,
        };

        match grouped.entry(candidate.canonical_id.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let total_records = entry.get().collapsed_records + 1;
                if should_replace_reprocess_candidate(entry.get(), &candidate) {
                    let mut replacement = candidate;
                    replacement.collapsed_records = total_records;
                    entry.insert(replacement);
                } else {
                    entry.get_mut().collapsed_records = total_records;
                }
            }
        }
    }

    let mut docs: Vec<ReprocessDocument> = grouped.into_values().collect();
    docs.sort_by(|left, right| left.canonical_id.cmp(&right.canonical_id));
    docs
}

fn reprocess_slice_mode_name(slice_mode: SliceMode) -> &'static str {
    match slice_mode {
        SliceMode::Onion => "onion",
        SliceMode::OnionFast => "onion-fast",
        SliceMode::Flat => "flat",
    }
}

fn prepare_reprocess_metadata(
    metadata: &Value,
    source_record_id: &str,
    source_text_hash: &str,
    collapsed_records: usize,
    slice_mode: SliceMode,
    source_label: &str,
    preprocess: bool,
) -> Value {
    let mut map = match metadata.clone() {
        Value::Object(map) => map,
        _ => Map::new(),
    };

    for key in [
        "layer",
        "parent_id",
        "children_ids",
        "original_id",
        "slice_mode",
        "content_hash",
    ] {
        map.remove(key);
    }

    map.insert(
        "slice_mode".to_string(),
        json!(reprocess_slice_mode_name(slice_mode)),
    );
    map.insert(
        "reprocess_source_record_id".to_string(),
        json!(source_record_id),
    );
    map.insert("reprocess_source_hash".to_string(), json!(source_text_hash));
    map.insert(
        "reprocess_collapsed_records".to_string(),
        json!(collapsed_records),
    );
    map.insert("reprocess_source".to_string(), json!(source_label));
    if preprocess {
        map.insert("reprocess_preprocessed".to_string(), json!(true));
    }

    Value::Object(map)
}

fn parse_export_records(content: &str) -> (Vec<ExportRecord>, usize) {
    let mut parse_errors = 0usize;
    let mut records = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<ExportRecord>(trimmed) {
            Ok(record) => records.push(record),
            Err(err) => {
                parse_errors += 1;
                eprintln!("  Line {}: parse error - {}", line_num + 1, err);
            }
        }
    }

    (records, parse_errors)
}

async fn run_reprocess_documents(
    plan: RebuildPlan,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let RebuildPlan {
        namespace,
        db_path,
        source_label,
        source_records,
        docs,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        parse_errors,
    } = plan;

    if docs.is_empty() {
        eprintln!("No non-empty documents found after collapsing export records");
        return Ok(());
    }

    let collapsed_records = source_records.saturating_sub(docs.len());

    eprintln!(
        "Reprocessing {} source records into {} canonical documents for namespace '{}'...",
        source_records,
        docs.len(),
        namespace
    );
    eprintln!("  Source:       {}", source_label);
    eprintln!("  Slice mode:   {}", reprocess_slice_mode_name(slice_mode));
    eprintln!(
        "  Preprocess:   {}",
        if preprocess { "enabled" } else { "disabled" }
    );
    eprintln!(
        "  Collapsed:    {} duplicate slice records",
        collapsed_records
    );
    if parse_errors > 0 {
        eprintln!("  Parse errors: {}", parse_errors);
    }

    if dry_run {
        eprintln!();
        eprintln!("Dry run only: no documents were written.");
        return Ok(());
    }

    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));

    {
        let mut guard = embedding_client.lock().await;
        guard.embed("healthcheck").await.map_err(|e| {
            anyhow!(
                "Embedding server preflight failed: {}. \
                 Fix the embedding server before reprocessing to avoid partial data loss.",
                e
            )
        })?;
        eprintln!("  Preflight:    embedding server OK");
    }

    let rag = RAGPipeline::new(embedding_client, storage).await?;
    let preprocessor = preprocess.then(|| Preprocessor::new(PreprocessingConfig::default()));
    let min_length = PreprocessingConfig::default().min_content_length;

    let mut indexed = 0usize;
    let mut replaced = 0usize;
    let mut skipped_existing_count = 0usize;
    let mut skipped_empty_input = 0usize;
    let mut skipped_preprocess_short = 0usize;
    let mut failed_ids: Vec<String> = Vec::new();
    let total = docs.len();

    for (idx, doc) in docs.iter().enumerate() {
        let existing = rag.lookup_memory(&namespace, &doc.canonical_id).await?;
        if let Some(existing_doc) = existing.as_ref()
            && skip_existing
            && existing_doc
                .metadata
                .get("reprocess_source_hash")
                .and_then(Value::as_str)
                == Some(doc.source_text_hash.as_str())
        {
            skipped_existing_count += 1;
            if (idx + 1) % 250 == 0 {
                eprintln!(
                    "  Progress: {}/{} (indexed:{} skipped:{})",
                    idx + 1,
                    total,
                    indexed,
                    skipped_existing_count
                );
            }
            continue;
        }

        let text = if let Some(preprocessor) = &preprocessor {
            preprocessor.extract_semantic_content(&doc.text)
        } else {
            doc.text.clone()
        };

        if text.trim().is_empty() {
            skipped_empty_input += 1;
            continue;
        }

        if preprocess && text.trim().len() < min_length {
            skipped_preprocess_short += 1;
            continue;
        }

        let metadata = prepare_reprocess_metadata(
            &doc.metadata,
            &doc.source_record_id,
            &doc.source_text_hash,
            doc.collapsed_records,
            slice_mode,
            &source_label,
            preprocess,
        );

        if existing.is_some() {
            replaced += 1;
        }

        match rag
            .memory_upsert(&namespace, doc.canonical_id.clone(), text, metadata)
            .await
        {
            Ok(()) => {
                indexed += 1;
            }
            Err(e) => {
                eprintln!("  WARN: failed to index '{}': {}", doc.canonical_id, e);
                failed_ids.push(doc.canonical_id.clone());
            }
        }

        if (idx + 1) % 250 == 0 {
            eprintln!(
                "  Progress: {}/{} (indexed:{} skipped:{} failed:{})",
                idx + 1,
                total,
                indexed,
                skipped_existing_count,
                failed_ids.len()
            );
        }
    }

    eprintln!();
    eprintln!("Reprocess complete:");
    eprintln!("  Indexed:         {}", indexed);
    if replaced > 0 {
        eprintln!("  Replaced:        {}", replaced);
    }
    if skipped_existing_count > 0 {
        eprintln!("  Skipped existing: {}", skipped_existing_count);
    }
    if skipped_empty_input > 0 {
        eprintln!("  Skipped empty:   {}", skipped_empty_input);
    }
    if skipped_preprocess_short > 0 {
        eprintln!("  Skipped too short: {}", skipped_preprocess_short);
    }
    if !failed_ids.is_empty() {
        eprintln!(
            "  FAILED:          {} (IDs: {})",
            failed_ids.len(),
            if failed_ids.len() <= 10 {
                failed_ids.join(", ")
            } else {
                format!(
                    "{}... and {} more",
                    failed_ids[..10].join(", "),
                    failed_ids.len() - 10
                )
            }
        );
    }
    if parse_errors > 0 {
        eprintln!("  Parse errors:    {}", parse_errors);
    }

    Ok(())
}

/// Export a namespace to JSONL file for portable backup
pub async fn run_export(
    namespace: String,
    output: Option<PathBuf>,
    include_embeddings: bool,
    db_path: String,
) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;

    const PAGE_SIZE: usize = 5000;
    let mut docs: Vec<rmcp_memex::ChromaDocument> = Vec::new();
    let mut offset = 0;
    loop {
        let page = storage
            .all_documents_page(Some(&namespace), offset, PAGE_SIZE)
            .await?;
        let page_len = page.len();
        if page_len > 0 {
            eprintln!(
                "  Loading: fetched {} documents (offset {})",
                page_len, offset
            );
        }
        docs.extend(page);
        if page_len < PAGE_SIZE {
            break;
        }
        offset += page_len;
    }

    if docs.is_empty() {
        eprintln!("No documents found in namespace '{}'", namespace);
        return Ok(());
    }

    eprintln!(
        "Exporting {} documents from namespace '{}'...",
        docs.len(),
        namespace
    );

    // Build JSONL output - each document on a separate line
    let mut lines: Vec<String> = Vec::with_capacity(docs.len());

    for doc in &docs {
        let record = ExportRecord {
            id: doc.id.clone(),
            text: doc.document.clone(),
            metadata: doc.metadata.clone(),
            content_hash: doc.content_hash.clone(),
            embeddings: if include_embeddings {
                Some(doc.embedding.clone())
            } else {
                None
            },
        };

        let line = serde_json::to_string(&record)?;
        lines.push(line);
    }

    let jsonl_content = lines.join("\n");

    match output {
        Some(path) => {
            tokio::fs::write(&path, &jsonl_content).await?;
            eprintln!(
                "Exported {} documents from '{}' to {:?}",
                docs.len(),
                namespace,
                path
            );
            if include_embeddings {
                eprintln!("  (embeddings included - file may be large)");
            }
        }
        None => {
            println!("{}", jsonl_content);
        }
    }

    Ok(())
}

/// Import documents from JSONL file into a namespace
pub async fn run_import(
    namespace: String,
    input: PathBuf,
    skip_existing: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let (_validated_input, content) = path_utils::safe_read_to_string_async(&input).await?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.is_empty() {
        eprintln!("No records found in input file");
        return Ok(());
    }

    eprintln!(
        "Importing {} records into namespace '{}'...",
        lines.len(),
        namespace
    );

    // Initialize storage and embedding client
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));

    let mut imported_count = 0usize;
    let mut skipped_count = 0usize;
    let mut error_count = 0usize;

    // Collect records that need embedding
    let mut records_to_embed: Vec<(ExportRecord, usize)> = Vec::new();
    let mut records_with_embeddings: Vec<(ExportRecord, Vec<f32>)> = Vec::new();

    // Parse all records first
    for (line_num, line) in lines.iter().enumerate() {
        let record: ExportRecord = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  Line {}: parse error - {}", line_num + 1, e);
                error_count += 1;
                continue;
            }
        };

        // Check for duplicates if skip_existing is enabled
        if skip_existing
            && let Some(ref hash) = record.content_hash
            && storage.has_content_hash(&namespace, hash).await?
        {
            skipped_count += 1;
            continue;
        }

        // Check if record has embeddings
        if record.embeddings.is_some() {
            let emb = record
                .embeddings
                .clone()
                .ok_or_else(|| anyhow!("missing embeddings"))?;
            records_with_embeddings.push((record, emb));
        } else {
            records_to_embed.push((record, line_num));
        }
    }

    // Process records that already have embeddings
    if !records_with_embeddings.is_empty() {
        eprintln!(
            "  Storing {} records with existing embeddings...",
            records_with_embeddings.len()
        );

        let mut docs = Vec::new();
        for (record, embedding) in records_with_embeddings {
            let hash = record
                .content_hash
                .unwrap_or_else(|| compute_content_hash(&record.text));
            let doc = rmcp_memex::ChromaDocument::new_flat_with_hash(
                record.id,
                namespace.clone(),
                embedding,
                record.metadata,
                record.text,
                hash,
            );
            docs.push(doc);
        }

        storage.add_to_store(docs.clone()).await?;
        imported_count += docs.len();
    }

    // Process records that need embedding
    if !records_to_embed.is_empty() {
        eprintln!(
            "  Re-embedding {} records without embeddings...",
            records_to_embed.len()
        );

        // Batch embed texts
        let texts: Vec<String> = records_to_embed
            .iter()
            .map(|(r, _)| r.text.clone())
            .collect();
        let embeddings = embedding_client.lock().await.embed_batch(&texts).await?;

        let mut docs = Vec::new();
        for ((record, _line_num), embedding) in records_to_embed.into_iter().zip(embeddings) {
            let hash = record
                .content_hash
                .unwrap_or_else(|| compute_content_hash(&record.text));
            let doc = rmcp_memex::ChromaDocument::new_flat_with_hash(
                record.id,
                namespace.clone(),
                embedding,
                record.metadata,
                record.text,
                hash,
            );
            docs.push(doc);
        }

        storage.add_to_store(docs.clone()).await?;
        imported_count += docs.len();
    }

    eprintln!();
    eprintln!("Import complete:");
    eprintln!("  Imported: {} documents", imported_count);
    if skipped_count > 0 {
        eprintln!("  Skipped:  {} (already exist)", skipped_count);
    }
    if error_count > 0 {
        eprintln!("  Errors:   {}", error_count);
    }

    Ok(())
}

/// Rebuild exported JSONL documents into a fresh namespace with current chunking.
pub async fn run_reprocess(
    config: ReprocessConfig,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let ReprocessConfig {
        namespace,
        input,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        db_path,
    } = config;

    let (_validated_input, content) = path_utils::safe_read_to_string_async(&input).await?;
    let (records, parse_errors) = parse_export_records(&content);

    if records.is_empty() {
        eprintln!("No valid export records found in {:?}", input);
        return Ok(());
    }

    let source_records = records.len();
    let docs = collapse_export_records(records);
    let source_label = input.display().to_string();

    run_reprocess_documents(
        RebuildPlan {
            namespace,
            db_path,
            source_label,
            source_records,
            docs,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            parse_errors,
        },
        embedding_config,
    )
    .await
}

pub async fn run_reindex(config: ReindexConfig, embedding_config: &EmbeddingConfig) -> Result<()> {
    let ReindexConfig {
        source_namespace,
        target_namespace,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        db_path,
    } = config;

    if source_namespace == target_namespace {
        return Err(anyhow!(
            "Source and target namespace are the same ('{}'). \
             In-place reindex risks data loss if the embedding server fails mid-operation. \
             Use a different target namespace, or pass --force-in-place if you have a backup.",
            source_namespace
        ));
    }

    let storage = StorageManager::new_lance_only(&db_path).await?;

    if !dry_run {
        let target_count = storage.count_namespace(&target_namespace).await?;
        if target_count > 0 && !skip_existing {
            return Err(anyhow!(
                "Target namespace '{}' already contains {} documents. \
                 Pass --skip-existing to resume, or use a fresh target namespace.",
                target_namespace,
                target_count
            ));
        }
    }

    const PAGE_SIZE: usize = 5000;
    let mut records: Vec<ExportRecord> = Vec::new();
    let mut offset = 0;

    loop {
        let page = storage
            .all_documents_page(Some(&source_namespace), offset, PAGE_SIZE)
            .await?;
        let page_len = page.len();
        eprintln!(
            "  Loading source: fetched {} documents (offset {})",
            page_len, offset
        );

        for doc in page {
            records.push(ExportRecord {
                id: doc.id,
                text: doc.document,
                metadata: doc.metadata,
                content_hash: doc.content_hash,
                embeddings: None,
            });
        }

        if page_len < PAGE_SIZE {
            break;
        }
        offset += page_len;
    }

    if records.is_empty() {
        eprintln!("No documents found in namespace '{}'", source_namespace);
        return Ok(());
    }

    let source_records = records.len();
    let collapsed = collapse_export_records(records);
    let source_label = format!("namespace:{}@{}", source_namespace, db_path);

    run_reprocess_documents(
        RebuildPlan {
            namespace: target_namespace,
            db_path,
            source_label,
            source_records,
            docs: collapsed,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            parse_errors: 0,
        },
        embedding_config,
    )
    .await
}

// =============================================================================
// AUDIT & PURGE QUALITY COMMANDS
// =============================================================================

/// Namespace audit result with quality metrics
#[derive(Debug, Serialize)]
pub struct NamespaceAuditResult {
    pub namespace: String,
    pub document_count: usize,
    pub avg_chunk_length: usize,
    pub sentence_integrity: f32,
    pub word_integrity: f32,
    pub chunk_quality: f32,
    pub overall_score: f32,
    pub recommendation: String,
    pub passes_threshold: bool,
}

/// Run audit on namespaces to check quality metrics
pub async fn run_audit(
    namespace: Option<String>,
    threshold: u8,
    verbose: bool,
    json: bool,
    db_path: String,
) -> Result<()> {
    use rmcp_memex::{IntegrityRecommendation, TextIntegrityMetrics};

    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Get namespaces to audit (list_namespaces returns Vec<(String, usize)>)
    let namespaces: Vec<String> = if let Some(ns) = namespace {
        vec![ns]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _count)| name)
            .collect()
    };

    if namespaces.is_empty() {
        if json {
            println!(r#"{{"namespaces": [], "summary": {{"total": 0}}}}"#);
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    let threshold_f32 = threshold as f32 / 100.0;
    let mut results: Vec<NamespaceAuditResult> = Vec::new();

    if !json {
        eprintln!(
            "Auditing {} namespace(s) with {}% quality threshold...\n",
            namespaces.len(),
            threshold
        );
    }

    for ns in &namespaces {
        // Get all documents in namespace
        let docs = storage.get_all_in_namespace(ns).await?;

        if docs.is_empty() {
            results.push(NamespaceAuditResult {
                namespace: ns.clone(),
                document_count: 0,
                avg_chunk_length: 0,
                sentence_integrity: 0.0,
                word_integrity: 0.0,
                chunk_quality: 0.0,
                overall_score: 0.0,
                recommendation: "EMPTY".to_string(),
                passes_threshold: false,
            });
            continue;
        }

        // Extract text from documents and compute metrics (ChromaDocument has `document` field)
        let chunks: Vec<String> = docs.iter().map(|d| d.document.clone()).collect();
        let combined_text = chunks.join(" ");

        let metrics = TextIntegrityMetrics::compute(&combined_text, &chunks);
        let passes = metrics.overall >= threshold_f32;

        let recommendation = match metrics.recommendation() {
            IntegrityRecommendation::Excellent => "EXCELLENT",
            IntegrityRecommendation::Good => "GOOD",
            IntegrityRecommendation::Warn => "WARN",
            IntegrityRecommendation::Purge => "PURGE",
        };

        results.push(NamespaceAuditResult {
            namespace: ns.clone(),
            document_count: docs.len(),
            avg_chunk_length: metrics.avg_chunk_length,
            sentence_integrity: metrics.sentence_integrity,
            word_integrity: metrics.word_integrity,
            chunk_quality: metrics.chunk_quality,
            overall_score: metrics.overall,
            recommendation: recommendation.to_string(),
            passes_threshold: passes,
        });

        if verbose && !json {
            eprintln!("Namespace: {}", ns);
            eprintln!("  Documents: {}", docs.len());
            eprintln!("  {}", metrics);
            eprintln!();
        }
    }

    // Output results
    if json {
        let passing = results.iter().filter(|r| r.passes_threshold).count();
        let failing = results.len() - passing;

        let output = serde_json::json!({
            "namespaces": results,
            "summary": {
                "total": results.len(),
                "passing": passing,
                "failing": failing,
                "threshold": threshold
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Human-readable output
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    NAMESPACE QUALITY AUDIT                     ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║ {:30} │ {:>6} │ {:>6} │ {:>8} ║",
            "Namespace", "Docs", "Score", "Status"
        );
        println!("╠════════════════════════════════════════════════════════════════╣");

        for result in &results {
            let status_icon = if result.passes_threshold {
                "✅"
            } else {
                "❌"
            };
            let ns_display = if result.namespace.len() > 28 {
                format!("{}...", &result.namespace[..25])
            } else {
                result.namespace.clone()
            };

            println!(
                "║ {:30} │ {:>6} │ {:>5.1}% │ {} {:>6} ║",
                ns_display,
                result.document_count,
                result.overall_score * 100.0,
                status_icon,
                result.recommendation
            );
        }

        println!("╚════════════════════════════════════════════════════════════════╝");

        let passing = results.iter().filter(|r| r.passes_threshold).count();
        let failing = results.len() - passing;

        println!();
        println!(
            "Summary: {} passing, {} failing (threshold: {}%)",
            passing, failing, threshold
        );

        if failing > 0 {
            println!();
            println!("Namespaces below threshold:");
            for result in results.iter().filter(|r| !r.passes_threshold) {
                println!(
                    "  - {} ({:.1}% quality, {} docs)",
                    result.namespace,
                    result.overall_score * 100.0,
                    result.document_count
                );
            }
            println!();
            println!(
                "Run 'rmcp-memex purge-quality --threshold {}' to remove low-quality namespaces",
                threshold
            );
        }
    }

    Ok(())
}

/// Purge namespaces below quality threshold
pub async fn run_purge_quality(
    threshold: u8,
    confirm: bool,
    json: bool,
    db_path: String,
) -> Result<()> {
    use rmcp_memex::TextIntegrityMetrics;

    let storage = StorageManager::new_lance_only(&db_path).await?;
    // list_namespaces returns Vec<(String, usize)>
    let namespace_list = storage.list_namespaces().await?;

    if namespace_list.is_empty() {
        if json {
            println!(r#"{{"purged": [], "dry_run": {}}}"#, !confirm);
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    let threshold_f32 = threshold as f32 / 100.0;
    let mut to_purge: Vec<(String, f32, usize)> = Vec::new();

    if !json {
        eprintln!(
            "Analyzing {} namespace(s) with {}% quality threshold...\n",
            namespace_list.len(),
            threshold
        );
    }

    for (ns, _count) in &namespace_list {
        let docs = storage.get_all_in_namespace(ns).await?;

        if docs.is_empty() {
            to_purge.push((ns.clone(), 0.0, 0));
            continue;
        }

        // ChromaDocument has `document` field, not `text`
        let chunks: Vec<String> = docs.iter().map(|d| d.document.clone()).collect();
        let combined_text = chunks.join(" ");
        let metrics = TextIntegrityMetrics::compute(&combined_text, &chunks);

        if metrics.overall < threshold_f32 {
            to_purge.push((ns.clone(), metrics.overall, docs.len()));
        }
    }

    if to_purge.is_empty() {
        if json {
            println!(r#"{{"purged": [], "message": "All namespaces pass quality threshold"}}"#);
        } else {
            println!(
                "All namespaces pass the {}% quality threshold. Nothing to purge.",
                threshold
            );
        }
        return Ok(());
    }

    if json {
        let output = serde_json::json!({
            "dry_run": !confirm,
            "threshold": threshold,
            "to_purge": to_purge.iter().map(|(ns, score, count)| {
                serde_json::json!({
                    "namespace": ns,
                    "quality_score": score,
                    "document_count": count
                })
            }).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&output)?);

        if !confirm {
            return Ok(());
        }
    } else {
        println!(
            "Found {} namespace(s) below {}% quality threshold:",
            to_purge.len(),
            threshold
        );
        for (ns, score, count) in &to_purge {
            println!("  - {} ({:.1}% quality, {} docs)", ns, score * 100.0, count);
        }
        println!();

        if !confirm {
            println!("DRY RUN - No changes made.");
            println!("Run with --confirm to actually delete these namespaces.");
            return Ok(());
        }
    }

    // Actually purge if confirmed (use purge_namespace, not delete_namespace)
    let mut purged_count = 0;
    for (ns, _score, count) in &to_purge {
        if !json {
            eprint!("Purging '{}' ({} docs)... ", ns, count);
        }

        match storage.delete_namespace_documents(ns).await {
            Ok(_) => {
                purged_count += 1;
                if !json {
                    eprintln!("done");
                }
            }
            Err(e) => {
                if !json {
                    eprintln!("ERROR: {}", e);
                }
            }
        }
    }

    if !json {
        println!();
        println!(
            "Purged {} namespace(s) with quality below {}%",
            purged_count, threshold
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_export_records_prefers_core_slice_for_same_original_id() {
        let records = vec![
            ExportRecord {
                id: "outer-1".to_string(),
                text: "Short summary".to_string(),
                metadata: json!({
                    "original_id": "doc-1",
                    "layer": "outer",
                    "project": "vista"
                }),
                content_hash: Some("outer-hash".to_string()),
                embeddings: None,
            },
            ExportRecord {
                id: "core-1".to_string(),
                text: "Longer full document content that should win during reprocess collapse."
                    .to_string(),
                metadata: json!({
                    "original_id": "doc-1",
                    "layer": "core",
                    "project": "vista"
                }),
                content_hash: Some("core-hash".to_string()),
                embeddings: None,
            },
        ];

        let docs = collapse_export_records(records);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].canonical_id, "doc-1");
        assert_eq!(docs[0].source_record_id, "core-1");
        assert_eq!(docs[0].collapsed_records, 2);
        assert!(docs[0].text.contains("full document content"));
    }

    #[test]
    fn collapse_export_records_falls_back_to_doc_id_then_record_id() {
        let records = vec![
            ExportRecord {
                id: "record-a".to_string(),
                text: "alpha".to_string(),
                metadata: json!({"doc_id": "doc-a"}),
                content_hash: None,
                embeddings: None,
            },
            ExportRecord {
                id: "record-b".to_string(),
                text: "beta".to_string(),
                metadata: json!({}),
                content_hash: None,
                embeddings: None,
            },
        ];

        let docs = collapse_export_records(records);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].canonical_id, "doc-a");
        assert_eq!(docs[1].canonical_id, "record-b");
    }

    #[test]
    fn prepare_reprocess_metadata_replaces_stale_chunk_fields() {
        let metadata = json!({
            "original_id": "old-doc",
            "layer": "outer",
            "slice_mode": "onion",
            "content_hash": "stale",
            "project": "vista"
        });

        let prepared = prepare_reprocess_metadata(
            &metadata,
            "core-1",
            "fresh-hash",
            4,
            SliceMode::OnionFast,
            "legacy.jsonl",
            true,
        );

        assert_eq!(prepared["project"], "vista");
        assert_eq!(prepared["slice_mode"], "onion-fast");
        assert_eq!(prepared["reprocess_source_record_id"], "core-1");
        assert_eq!(prepared["reprocess_source_hash"], "fresh-hash");
        assert_eq!(prepared["reprocess_collapsed_records"], 4);
        assert_eq!(prepared["reprocess_source"], "legacy.jsonl");
        assert_eq!(prepared["reprocess_preprocessed"], true);
        assert!(prepared.get("layer").is_none());
        assert!(prepared.get("original_id").is_none());
        assert!(prepared.get("content_hash").is_none());
    }

    #[test]
    fn default_reindexed_namespace_appends_suffix() {
        assert_eq!(
            default_reindexed_namespace("kodowanie"),
            "kodowanie-reindexed"
        );
    }
}
