//! Indexer event contracts shared by the scheduler, TUI, and future consumers.

use std::collections::VecDeque;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

/// Events emitted by the indexing scheduler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexEvent {
    RunStarted {
        total_files: usize,
        namespace: String,
        source_dir: String,
        parallelism: usize,
        started_at: DateTime<Utc>,
    },
    FileStarted {
        file_index: usize,
        path: String,
        size_bytes: u64,
    },
    FileIndexed {
        file_index: usize,
        path: String,
        chunks_indexed: usize,
        content_hash: String,
        duration_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        embedder_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens_estimated: Option<usize>,
    },
    FileSkipped {
        file_index: usize,
        path: String,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content_hash: Option<String>,
    },
    FileFailed {
        file_index: usize,
        path: String,
        error: String,
    },
    StatsTick {
        processed: usize,
        indexed: usize,
        skipped: usize,
        failed: usize,
        total: usize,
        files_per_sec: f64,
        eta_secs: Option<f64>,
        total_chunks: usize,
        in_flight: usize,
    },
    RunCompleted {
        processed: usize,
        indexed: usize,
        skipped: usize,
        failed: usize,
        total_chunks: usize,
        elapsed: Duration,
        stopped_early: bool,
    },
    RunFailed {
        error: String,
        processed_before_failure: usize,
    },
    Paused,
    Resumed,
    ParallelismChanged {
        previous: usize,
        current: usize,
    },
    StopRequested,
    Warning {
        code: String,
        message: String,
    },
}

/// Event sinks must stay synchronous and infallible.
pub trait IndexEventSink: Send + Sync {
    fn on_event(&self, event: &IndexEvent);
}

/// Maximum number of recent warnings kept in the live snapshot.
pub const MAX_RECENT_WARNINGS: usize = 20;

/// Pull-friendly indexing telemetry snapshot for the TUI and future tooling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexTelemetrySnapshot {
    pub namespace: String,
    pub source_dir: String,
    pub started_at: Option<DateTime<Utc>>,
    pub total: usize,
    pub processed: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub total_chunks: usize,
    pub current_file: Option<String>,
    pub in_flight: usize,
    pub parallelism: usize,
    pub paused: bool,
    pub stopping: bool,
    pub files_per_sec: f64,
    pub eta_secs: Option<f64>,
    pub elapsed: Duration,
    pub avg_embedder_ms: Option<f64>,
    pub total_tokens_estimated: usize,
    pub complete: bool,
    pub stopped_early: bool,
    pub fatal_error: Option<String>,
    pub recent_warnings: VecDeque<WarningEntry>,
}

impl Default for IndexTelemetrySnapshot {
    fn default() -> Self {
        Self {
            namespace: String::new(),
            source_dir: String::new(),
            started_at: None,
            total: 0,
            processed: 0,
            indexed: 0,
            skipped: 0,
            failed: 0,
            total_chunks: 0,
            current_file: None,
            in_flight: 0,
            parallelism: 1,
            paused: false,
            stopping: false,
            files_per_sec: 0.0,
            eta_secs: None,
            elapsed: Duration::ZERO,
            avg_embedder_ms: None,
            total_tokens_estimated: 0,
            complete: false,
            stopped_early: false,
            fatal_error: None,
            recent_warnings: VecDeque::new(),
        }
    }
}

/// Warning displayed in the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WarningEntry {
    pub code: String,
    pub message: String,
    pub at: DateTime<Utc>,
}

/// Shared watch sender for the latest telemetry snapshot.
///
/// This mirrors the rmcp-mux `publish_status` pattern: writers can emit
/// frequently while consumers always observe the newest coalesced snapshot.
pub type SharedIndexTelemetry = watch::Sender<IndexTelemetrySnapshot>;

/// Build the watch channel used by the dashboard.
pub fn new_index_telemetry() -> (
    SharedIndexTelemetry,
    watch::Receiver<IndexTelemetrySnapshot>,
) {
    watch::channel(IndexTelemetrySnapshot::default())
}

/// Control messages sent from the TUI to the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexControl {
    Pause,
    Resume,
    SetParallelism(usize),
    Stop,
}

/// Bounded control channel size.
pub const INDEX_CONTROL_CHANNEL_CAPACITY: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_event_serde_roundtrip_representative_variants() {
        let events = vec![
            IndexEvent::RunStarted {
                total_files: 12,
                namespace: "kb:test".to_string(),
                source_dir: "/tmp/input".to_string(),
                parallelism: 4,
                started_at: Utc::now(),
            },
            IndexEvent::FileStarted {
                file_index: 2,
                path: "notes.md".to_string(),
                size_bytes: 512,
            },
            IndexEvent::FileIndexed {
                file_index: 2,
                path: "notes.md".to_string(),
                chunks_indexed: 7,
                content_hash: "abc123".to_string(),
                duration_ms: 231,
                embedder_ms: Some(187),
                tokens_estimated: Some(128),
            },
            IndexEvent::StatsTick {
                processed: 8,
                indexed: 6,
                skipped: 1,
                failed: 1,
                total: 12,
                files_per_sec: 1.5,
                eta_secs: Some(2.6),
                total_chunks: 18,
                in_flight: 2,
            },
            IndexEvent::RunCompleted {
                processed: 12,
                indexed: 9,
                skipped: 2,
                failed: 1,
                total_chunks: 28,
                elapsed: Duration::from_secs(12),
                stopped_early: false,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).expect("serialize event");
            let roundtrip: IndexEvent = serde_json::from_str(&json).expect("deserialize event");
            assert_eq!(roundtrip, event);
        }
    }
}
