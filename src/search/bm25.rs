//! BM25 keyword search using Tantivy.
//!
//! Provides exact keyword matching to complement vector similarity search.
//! This helps distinguish between semantically similar but distinct terms
//! like "smutny" (sad) and "melancholijny" (melancholic).
//!
//! Lock strategy: On-demand IndexWriter acquisition/release per write batch.
//! This allows multiple processes to write sequentially without permanent lock holding.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tantivy::{
    Index, IndexReader, TantivyDocument,
    collector::TopDocs,
    query::{AllQuery, QueryParser},
    schema::{
        Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
    },
    tokenizer::{Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer},
};
use tokio::sync::Mutex;

/// Supported languages for stemming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StemLanguage {
    #[default]
    English,
    German,
    French,
    Spanish,
    Italian,
    Portuguese,
    Russian,
    /// No stemming (for unsupported languages like Polish)
    None,
}

impl StemLanguage {
    fn to_tantivy_language(self) -> Option<Language> {
        match self {
            StemLanguage::English => Some(Language::English),
            StemLanguage::German => Some(Language::German),
            StemLanguage::French => Some(Language::French),
            StemLanguage::Spanish => Some(Language::Spanish),
            StemLanguage::Italian => Some(Language::Italian),
            StemLanguage::Portuguese => Some(Language::Portuguese),
            StemLanguage::Russian => Some(Language::Russian),
            StemLanguage::None => None,
        }
    }
}

/// Configuration for BM25 index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// Path to store the Tantivy index
    #[serde(default = "default_bm25_path")]
    pub index_path: String,
    /// Heap size for index writer (bytes)
    #[serde(default = "default_heap_size")]
    pub writer_heap_size: usize,
    /// Enable stemming for better recall
    #[serde(default = "default_true")]
    pub enable_stemming: bool,
    /// Language for stemming
    #[serde(default)]
    pub language: StemLanguage,
    /// Read-only mode - disables write operations entirely
    /// Use for dedicated read-only instances
    #[serde(default)]
    pub read_only: bool,
}

fn default_bm25_path() -> String {
    "~/.rmcp-servers/rmcp-memex/bm25".to_string()
}

fn default_heap_size() -> usize {
    50_000_000
}

fn default_true() -> bool {
    true
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            index_path: default_bm25_path(),
            writer_heap_size: default_heap_size(),
            enable_stemming: true,
            language: StemLanguage::English,
            read_only: false,
        }
    }
}

impl BM25Config {
    /// Create config for multilingual content (no stemming)
    pub fn multilingual() -> Self {
        Self {
            language: StemLanguage::None,
            enable_stemming: false,
            ..Self::default()
        }
    }

    /// Create read-only config (disables write operations)
    pub fn read_only() -> Self {
        Self {
            read_only: true,
            ..Self::default()
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.index_path = path.into();
        self
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

/// BM25 keyword search index using Tantivy
///
/// Uses on-demand IndexWriter acquisition: lock acquired only during writes,
/// released immediately after commit. This allows multiple processes to write
/// sequentially without permanent lock holding.
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    content_field: Field,
    id_field: Field,
    namespace_field: Field,
    /// Heap size for writer (used when acquiring on-demand)
    writer_heap_size: usize,
    /// Read-only mode flag
    read_only: bool,
    /// Mutex to serialize write operations within this process
    write_lock: Arc<Mutex<()>>,
    /// Index path for error messages
    index_path: PathBuf,
}

impl BM25Index {
    /// Create or open a BM25 index at the given path
    pub fn new(config: &BM25Config) -> Result<Self> {
        let path = crate::path_utils::sanitize_new_path(&config.index_path)?;

        // Create directory if it doesn't exist
        if !path.exists() {
            std::fs::create_dir_all(&path)?;
        }

        // Build schema with text analysis
        let mut schema_builder = Schema::builder();

        // Configure text field with proper tokenization
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("custom_tokenizer")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let content_field = schema_builder.add_text_field("content", text_options);
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let namespace_field = schema_builder.add_text_field("namespace", STRING | STORED);

        let schema = schema_builder.build();

        // Open or create index
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(&path)?
        } else {
            Index::create_in_dir(&path, schema.clone())?
        };

        // Register custom tokenizer with optional stemming
        let tokenizer = if config.enable_stemming {
            if let Some(lang) = config.language.to_tantivy_language() {
                TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(RemoveLongFilter::limit(40))
                    .filter(LowerCaser)
                    .filter(Stemmer::new(lang))
                    .build()
            } else {
                // No stemming for unsupported languages
                TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(RemoveLongFilter::limit(40))
                    .filter(LowerCaser)
                    .build()
            }
        } else {
            TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build()
        };

        index.tokenizers().register("custom_tokenizer", tokenizer);

        let reader = index.reader()?;

        if config.read_only {
            tracing::info!("BM25 index opened in READ-ONLY mode");
        } else {
            tracing::debug!("BM25 index opened (on-demand lock acquisition for writes)");
        }

        Ok(Self {
            index,
            reader,
            content_field,
            id_field,
            namespace_field,
            writer_heap_size: config.writer_heap_size,
            read_only: config.read_only,
            write_lock: Arc::new(Mutex::new(())),
            index_path: path,
        })
    }

    /// Check if index is in read-only mode
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Acquire IndexWriter, perform write operation, release lock
    ///
    /// This is the core pattern: acquire lock -> write -> commit -> drop (release)
    /// Includes retry with exponential backoff for lock contention.
    async fn with_writer<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce(&mut tantivy::IndexWriter) -> Result<T>,
    {
        if self.read_only {
            return Err(anyhow!("Cannot write: BM25 index is in read-only mode"));
        }

        // Serialize writes within this process
        let _guard = self.write_lock.lock().await;

        // Retry with exponential backoff for cross-process lock contention
        const MAX_RETRIES: u32 = 5;
        const INITIAL_BACKOFF_MS: u64 = 50;
        const MAX_BACKOFF_MS: u64 = 2000;

        let mut attempt = 0;
        let mut backoff_ms = INITIAL_BACKOFF_MS;

        let mut writer = loop {
            match self.index.writer(self.writer_heap_size) {
                Ok(w) => break w,
                Err(e) => {
                    let is_lock_busy = e.to_string().contains("LockBusy");

                    if is_lock_busy && attempt < MAX_RETRIES {
                        attempt += 1;
                        tracing::debug!(
                            "BM25 lock busy, retry {}/{} in {}ms. Path: {:?}",
                            attempt,
                            MAX_RETRIES,
                            backoff_ms,
                            self.index_path
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                    } else if is_lock_busy {
                        return Err(anyhow!(
                            "BM25 index locked after {} retries. Path: {:?}. \
                             Multiple processes writing simultaneously - try again.",
                            MAX_RETRIES,
                            self.index_path
                        ));
                    } else {
                        return Err(anyhow!("Failed to acquire BM25 writer: {}", e));
                    }
                }
            }
        };

        // Perform the write operation
        let result = operation(&mut writer)?;

        // Commit changes
        writer.commit()?;

        // Writer dropped here -> Tantivy lock released
        drop(writer);

        // Reload reader to see new data
        self.reader.reload()?;

        Ok(result)
    }

    /// Add documents to the BM25 index
    ///
    /// Lock is acquired only for the duration of this operation.
    ///
    /// # Arguments
    /// * `docs` - List of (id, namespace, content) tuples
    ///
    /// # Errors
    /// Returns error if index is in read-only mode or another process holds the lock
    pub async fn add_documents(&self, docs: &[(String, String, String)]) -> Result<()> {
        let content_field = self.content_field;
        let id_field = self.id_field;
        let namespace_field = self.namespace_field;
        let doc_count = docs.len();

        // Clone docs for the closure (needed because closure must be 'static for FnOnce)
        let docs = docs.to_vec();

        self.with_writer(move |writer| {
            for (id, namespace, content) in &docs {
                let mut doc = TantivyDocument::new();
                doc.add_text(content_field, content);
                doc.add_text(id_field, id);
                doc.add_text(namespace_field, namespace);
                writer.add_document(doc)?;
            }
            Ok(())
        })
        .await?;

        tracing::debug!("Added {} documents to BM25 index", doc_count);
        Ok(())
    }

    /// Search the BM25 index
    ///
    /// # Arguments
    /// * `query` - Search query string
    /// * `namespace` - Optional namespace filter
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Vector of (document_id, namespace, score) tuples, sorted by score descending
    pub fn search(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, String, f32)>> {
        let searcher = self.reader.searcher();

        // Build query - search in content field
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        // Escape special characters and parse query
        let escaped_query = Self::escape_query(query);
        let parsed_query = query_parser
            .parse_query(&escaped_query)
            .map_err(|e| anyhow!("Query parse error: {}", e))?;

        // Execute search
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit * 2))?;

        let mut results = Vec::with_capacity(limit);

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            // Get document ID and namespace using stored fields.
            let id = doc
                .get_first(self.id_field)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()))
                .ok_or_else(|| anyhow!("Document missing ID field"))?;
            let doc_namespace = doc
                .get_first(self.namespace_field)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()))
                .ok_or_else(|| anyhow!("Document missing namespace field"))?;

            // Filter by namespace if specified
            if let Some(ns) = namespace
                && doc_namespace != ns
            {
                continue;
            }

            results.push((id, doc_namespace, score));

            if results.len() >= limit {
                break;
            }
        }

        tracing::debug!("BM25 search '{}' returned {} results", query, results.len());

        Ok(results)
    }

    /// Delete documents by ID
    ///
    /// Lock is acquired only for the duration of this operation.
    ///
    /// # Errors
    /// Returns error if index is in read-only mode or another process holds the lock
    pub async fn delete_documents(&self, ids: &[String]) -> Result<usize> {
        let id_field = self.id_field;
        let ids = ids.to_vec();
        let count = ids.len();

        self.with_writer(move |writer| {
            for id in &ids {
                let term = tantivy::Term::from_field_text(id_field, id);
                writer.delete_term(term);
            }
            Ok(count)
        })
        .await
    }

    /// Delete all documents in a namespace
    ///
    /// Lock is acquired only for the duration of this operation.
    ///
    /// # Errors
    /// Returns error if index is in read-only mode or another process holds the lock
    pub async fn delete_namespace_term(&self, namespace: &str) -> Result<usize> {
        let namespace_field = self.namespace_field;
        let namespace_owned = namespace.to_string();
        let namespace_log = namespace.to_string();

        self.with_writer(move |writer| {
            let term = tantivy::Term::from_field_text(namespace_field, &namespace_owned);
            writer.delete_term(term);
            Ok(1) // Tantivy doesn't return exact count for term deletes
        })
        .await?;

        tracing::info!("Purged namespace '{}' from BM25 index", namespace_log);
        Ok(1)
    }

    /// Escape special query characters
    fn escape_query(query: &str) -> String {
        // Tantivy query syntax special characters
        let special_chars = [
            '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':',
            '\\', '/',
        ];

        let mut escaped = String::with_capacity(query.len() * 2);
        for c in query.chars() {
            if special_chars.contains(&c) {
                escaped.push('\\');
            }
            escaped.push(c);
        }
        escaped
    }

    /// Get document count in index
    pub fn doc_count(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }

    /// Return the stored `(namespace, id)` keys currently present in the index.
    ///
    /// This is intentionally an operator/debug surface used by recovery checks,
    /// not a hot-path query primitive.
    pub fn document_keys(&self, namespace: Option<&str>) -> Result<HashSet<(String, String)>> {
        let searcher = self.reader.searcher();
        let total = usize::try_from(searcher.num_docs()).unwrap_or(usize::MAX);
        if total == 0 {
            return Ok(HashSet::new());
        }

        let all_query = AllQuery;
        let top_docs = searcher.search(&all_query, &TopDocs::with_limit(total))?;
        let mut keys = HashSet::with_capacity(total);

        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let id = doc
                .get_first(self.id_field)
                .and_then(|value| Value::as_str(&value).map(|value| value.to_string()))
                .ok_or_else(|| anyhow!("Document missing ID field"))?;
            let doc_namespace = doc
                .get_first(self.namespace_field)
                .and_then(|value| Value::as_str(&value).map(|value| value.to_string()))
                .ok_or_else(|| anyhow!("Document missing namespace field"))?;

            if namespace.is_none_or(|expected| expected == doc_namespace) {
                keys.insert((doc_namespace, id));
            }
        }

        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_bm25_basic() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        // Add some documents
        let docs = vec![
            (
                "doc1".to_string(),
                "test".to_string(),
                "The quick brown fox jumps over the lazy dog".to_string(),
            ),
            (
                "doc2".to_string(),
                "test".to_string(),
                "A quick brown dog runs in the park".to_string(),
            ),
            (
                "doc3".to_string(),
                "test".to_string(),
                "The lazy cat sleeps all day".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();

        // Search
        let results = index.search("quick brown", None, 10).unwrap();

        assert_eq!(results.len(), 2);
        // doc1 and doc2 should match, doc3 should not
        let ids: Vec<&str> = results.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc2"));
    }

    #[tokio::test]
    async fn test_bm25_namespace_filter() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        let docs = vec![
            (
                "doc1".to_string(),
                "ns1".to_string(),
                "hello world".to_string(),
            ),
            (
                "doc2".to_string(),
                "ns2".to_string(),
                "hello universe".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();

        // Search with namespace filter
        let results = index.search("hello", Some("ns1"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1");
        assert_eq!(results[0].1, "ns1");
    }

    #[tokio::test]
    async fn test_bm25_delete_documents_removes_exact_id_matches() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        let docs = vec![
            (
                "doc1".to_string(),
                "team:alpha".to_string(),
                "shared search term".to_string(),
            ),
            (
                "doc2".to_string(),
                "team:alpha".to_string(),
                "shared search term".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();
        assert_eq!(index.search("shared", None, 10).unwrap().len(), 2);

        let deleted = index.delete_documents(&["doc1".to_string()]).await.unwrap();
        assert_eq!(deleted, 1);

        let results = index.search("shared", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc2");
    }

    #[tokio::test]
    async fn test_bm25_purge_namespace_matches_exact_string() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        let docs = vec![
            (
                "doc1".to_string(),
                "team:alpha".to_string(),
                "shared search term".to_string(),
            ),
            (
                "doc2".to_string(),
                "team:beta".to_string(),
                "shared search term".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();
        assert_eq!(index.search("shared", None, 10).unwrap().len(), 2);

        let deleted = index.delete_namespace_term("team:alpha").await.unwrap();
        assert_eq!(deleted, 1);

        assert!(
            index
                .search("shared", Some("team:alpha"), 10)
                .unwrap()
                .is_empty()
        );

        let remaining = index.search("shared", None, 10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].0, "doc2");
        assert_eq!(remaining[0].1, "team:beta");
    }

    #[tokio::test]
    async fn test_bm25_lock_release() {
        // Test that lock is released after write
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let config = BM25Config::default().with_path(path);
        let index1 = BM25Index::new(&config).unwrap();

        // First write - use "hello world" content
        index1
            .add_documents(&[(
                "doc1".to_string(),
                "ns".to_string(),
                "hello world".to_string(),
            )])
            .await
            .unwrap();

        // Drop first instance to ensure all resources released
        drop(index1);

        // Second instance should be able to write (lock released) and see committed data
        let config2 = BM25Config::default().with_path(path);
        let index2 = BM25Index::new(&config2).unwrap();

        // Use same keyword "hello" so both match
        index2
            .add_documents(&[(
                "doc2".to_string(),
                "ns".to_string(),
                "hello there".to_string(),
            )])
            .await
            .unwrap();

        // Both docs should be searchable with "hello"
        let results = index2.search("hello", None, 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_escape_query() {
        assert_eq!(BM25Index::escape_query("hello world"), "hello world");
        assert_eq!(BM25Index::escape_query("hello+world"), "hello\\+world");
        assert_eq!(BM25Index::escape_query("test:query"), "test\\:query");
    }
}
