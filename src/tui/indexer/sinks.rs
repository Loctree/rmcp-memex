//! Concrete index event sinks.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use chrono::Utc;

use super::contracts::{
    IndexEvent, IndexEventSink, IndexTelemetrySnapshot, MAX_RECENT_WARNINGS, SharedIndexTelemetry,
    WarningEntry,
};

const RATE_WINDOW_SIZE: usize = 50;

#[derive(Debug, Clone)]
struct CompletionSample {
    completed_at: Instant,
    embedder_ms: Option<u64>,
}

/// Updates the latest dashboard snapshot for the TUI.
pub struct TuiTelemetrySink {
    sender: Arc<SharedIndexTelemetry>,
    completion_window: StdMutex<VecDeque<CompletionSample>>,
    run_started_at: StdMutex<Option<Instant>>,
}

impl TuiTelemetrySink {
    pub fn new(sender: Arc<SharedIndexTelemetry>) -> Self {
        Self {
            sender,
            completion_window: StdMutex::new(VecDeque::with_capacity(RATE_WINDOW_SIZE)),
            run_started_at: StdMutex::new(None),
        }
    }

    fn completion_window_rate(window: &VecDeque<CompletionSample>) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }

        let oldest = window.front().map(|entry| entry.completed_at);
        let newest = window.back().map(|entry| entry.completed_at);
        match (oldest, newest) {
            (Some(oldest), Some(newest)) => {
                let seconds = newest.duration_since(oldest).as_secs_f64();
                if seconds <= f64::EPSILON {
                    0.0
                } else {
                    (window.len() - 1) as f64 / seconds
                }
            }
            _ => 0.0,
        }
    }

    fn average_embedder_ms(window: &VecDeque<CompletionSample>) -> Option<f64> {
        let mut total = 0_u64;
        let mut count = 0_u64;

        for sample in window {
            if let Some(embedder_ms) = sample.embedder_ms {
                total += embedder_ms;
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some(total as f64 / count as f64)
        }
    }

    fn touch_elapsed(&self, snapshot: &mut IndexTelemetrySnapshot) {
        let run_started_at = self
            .run_started_at
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(started_at) = *run_started_at {
            snapshot.elapsed = started_at.elapsed();
        }
    }

    fn recompute_rates(&self, snapshot: &mut IndexTelemetrySnapshot) {
        let window = self
            .completion_window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let rate = Self::completion_window_rate(&window);
        snapshot.files_per_sec = rate;
        snapshot.avg_embedder_ms = Self::average_embedder_ms(&window);
        if rate > f64::EPSILON {
            let remaining = snapshot.total.saturating_sub(snapshot.processed);
            snapshot.eta_secs = Some(remaining as f64 / rate);
        } else {
            snapshot.eta_secs = None;
        }
    }

    fn push_completion(&self, embedder_ms: Option<u64>) {
        let mut window = self
            .completion_window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if window.len() == RATE_WINDOW_SIZE {
            window.pop_front();
        }
        window.push_back(CompletionSample {
            completed_at: Instant::now(),
            embedder_ms,
        });
    }

    fn send_snapshot(&self, snapshot: IndexTelemetrySnapshot) {
        let _ = self.sender.send(snapshot);
    }
}

impl IndexEventSink for TuiTelemetrySink {
    fn on_event(&self, event: &IndexEvent) {
        let mut snapshot = self.sender.borrow().clone();

        match event {
            IndexEvent::RunStarted {
                total_files,
                namespace,
                source_dir,
                parallelism,
                started_at,
            } => {
                snapshot = IndexTelemetrySnapshot::default();
                snapshot.total = *total_files;
                snapshot.namespace = namespace.clone();
                snapshot.source_dir = source_dir.clone();
                snapshot.parallelism = *parallelism;
                snapshot.started_at = Some(*started_at);
                *self
                    .run_started_at
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Instant::now());
                self.completion_window
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clear();
            }
            IndexEvent::FileStarted { path, .. } => {
                snapshot.current_file = Some(path.clone());
                snapshot.in_flight += 1;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::FileIndexed {
                chunks_indexed,
                embedder_ms,
                tokens_estimated,
                ..
            } => {
                snapshot.processed += 1;
                snapshot.indexed += 1;
                snapshot.total_chunks += chunks_indexed;
                snapshot.in_flight = snapshot.in_flight.saturating_sub(1);
                snapshot.stopping = false;
                if let Some(tokens_estimated) = tokens_estimated {
                    snapshot.total_tokens_estimated += tokens_estimated;
                }
                if snapshot.in_flight == 0 {
                    snapshot.current_file = None;
                }
                self.push_completion(*embedder_ms);
                self.touch_elapsed(&mut snapshot);
                self.recompute_rates(&mut snapshot);
            }
            IndexEvent::FileSkipped { .. } => {
                snapshot.processed += 1;
                snapshot.skipped += 1;
                snapshot.in_flight = snapshot.in_flight.saturating_sub(1);
                snapshot.stopping = false;
                if snapshot.in_flight == 0 {
                    snapshot.current_file = None;
                }
                self.push_completion(None);
                self.touch_elapsed(&mut snapshot);
                self.recompute_rates(&mut snapshot);
            }
            IndexEvent::FileFailed { .. } => {
                snapshot.processed += 1;
                snapshot.failed += 1;
                snapshot.in_flight = snapshot.in_flight.saturating_sub(1);
                snapshot.stopping = false;
                if snapshot.in_flight == 0 {
                    snapshot.current_file = None;
                }
                self.push_completion(None);
                self.touch_elapsed(&mut snapshot);
                self.recompute_rates(&mut snapshot);
            }
            IndexEvent::StatsTick {
                processed,
                indexed,
                skipped,
                failed,
                total,
                total_chunks,
                in_flight,
                ..
            } => {
                snapshot.processed = *processed;
                snapshot.indexed = *indexed;
                snapshot.skipped = *skipped;
                snapshot.failed = *failed;
                snapshot.total = *total;
                snapshot.total_chunks = *total_chunks;
                snapshot.in_flight = *in_flight;
                self.touch_elapsed(&mut snapshot);
                self.recompute_rates(&mut snapshot);
            }
            IndexEvent::RunCompleted {
                processed,
                indexed,
                skipped,
                failed,
                total_chunks,
                elapsed,
                stopped_early,
            } => {
                snapshot.processed = *processed;
                snapshot.indexed = *indexed;
                snapshot.skipped = *skipped;
                snapshot.failed = *failed;
                snapshot.total_chunks = *total_chunks;
                snapshot.elapsed = *elapsed;
                snapshot.in_flight = 0;
                snapshot.current_file = None;
                snapshot.complete = true;
                snapshot.paused = false;
                snapshot.stopping = false;
                snapshot.stopped_early = *stopped_early;
                self.recompute_rates(&mut snapshot);
            }
            IndexEvent::RunFailed {
                error,
                processed_before_failure,
            } => {
                snapshot.processed = *processed_before_failure;
                snapshot.complete = true;
                snapshot.fatal_error = Some(error.clone());
                snapshot.current_file = None;
                snapshot.in_flight = 0;
                snapshot.paused = false;
                snapshot.stopping = false;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::Paused => {
                snapshot.paused = true;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::Resumed => {
                snapshot.paused = false;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::ParallelismChanged { current, .. } => {
                snapshot.parallelism = *current;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::StopRequested => {
                snapshot.stopping = true;
                snapshot.paused = false;
                self.touch_elapsed(&mut snapshot);
            }
            IndexEvent::Warning { code, message } => {
                if snapshot.recent_warnings.len() == MAX_RECENT_WARNINGS {
                    snapshot.recent_warnings.pop_front();
                }
                snapshot.recent_warnings.push_back(WarningEntry {
                    code: code.clone(),
                    message: message.clone(),
                    at: Utc::now(),
                });
                self.touch_elapsed(&mut snapshot);
            }
        }

        self.send_snapshot(snapshot);
    }
}

/// Emits tracing logs for indexing events.
pub struct TracingSink;

impl IndexEventSink for TracingSink {
    fn on_event(&self, event: &IndexEvent) {
        match event {
            IndexEvent::RunStarted {
                total_files,
                namespace,
                source_dir,
                parallelism,
                ..
            } => tracing::info!(
                total_files,
                namespace,
                source_dir,
                parallelism,
                "indexing run started"
            ),
            IndexEvent::FileStarted {
                file_index, path, ..
            } => tracing::debug!(file_index, path, "file indexing started"),
            IndexEvent::FileIndexed {
                file_index,
                path,
                chunks_indexed,
                duration_ms,
                ..
            } => tracing::info!(
                file_index,
                path,
                chunks_indexed,
                duration_ms,
                "file indexed"
            ),
            IndexEvent::FileSkipped {
                file_index,
                path,
                reason,
                ..
            } => tracing::debug!(file_index, path, reason, "file skipped"),
            IndexEvent::FileFailed {
                file_index,
                path,
                error,
            } => tracing::warn!(file_index, path, error, "file failed"),
            IndexEvent::StatsTick {
                processed,
                total,
                files_per_sec,
                in_flight,
                ..
            } => tracing::debug!(
                processed,
                total,
                files_per_sec,
                in_flight,
                "index stats tick"
            ),
            IndexEvent::RunCompleted {
                processed,
                indexed,
                skipped,
                failed,
                total_chunks,
                stopped_early,
                elapsed,
            } => tracing::info!(
                processed,
                indexed,
                skipped,
                failed,
                total_chunks,
                stopped_early,
                elapsed_secs = elapsed.as_secs_f64(),
                "indexing run completed"
            ),
            IndexEvent::RunFailed { error, .. } => {
                tracing::error!(error, "indexing run failed");
            }
            IndexEvent::Paused => tracing::info!("indexing paused"),
            IndexEvent::Resumed => tracing::info!("indexing resumed"),
            IndexEvent::ParallelismChanged { previous, current } => {
                tracing::info!(previous, current, "indexing parallelism changed");
            }
            IndexEvent::StopRequested => tracing::info!("indexing stop requested"),
            IndexEvent::Warning { code, message } => {
                tracing::warn!(code, message, "indexing warning");
            }
        }
    }
}

/// Forwards every event to every child sink.
pub struct FanOut {
    sinks: Vec<Arc<dyn IndexEventSink>>,
}

impl FanOut {
    pub fn new(sinks: Vec<Arc<dyn IndexEventSink>>) -> Self {
        Self { sinks }
    }
}

impl IndexEventSink for FanOut {
    fn on_event(&self, event: &IndexEvent) {
        for sink in &self.sinks {
            sink.on_event(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::indexer::contracts::{IndexEvent, new_index_telemetry};

    #[test]
    fn tui_telemetry_sink_tracks_completion_state() {
        let (sender, receiver) = new_index_telemetry();
        let sink = TuiTelemetrySink::new(Arc::new(sender));

        sink.on_event(&IndexEvent::RunStarted {
            total_files: 4,
            namespace: "kb:test".to_string(),
            source_dir: "/tmp/docs".to_string(),
            parallelism: 2,
            started_at: Utc::now(),
        });
        sink.on_event(&IndexEvent::FileStarted {
            file_index: 0,
            path: "a.md".to_string(),
            size_bytes: 10,
        });
        sink.on_event(&IndexEvent::FileIndexed {
            file_index: 0,
            path: "a.md".to_string(),
            chunks_indexed: 3,
            content_hash: "aaa".to_string(),
            duration_ms: 10,
            embedder_ms: Some(7),
            tokens_estimated: Some(20),
        });
        sink.on_event(&IndexEvent::FileStarted {
            file_index: 1,
            path: "b.md".to_string(),
            size_bytes: 11,
        });
        sink.on_event(&IndexEvent::FileFailed {
            file_index: 1,
            path: "b.md".to_string(),
            error: "boom".to_string(),
        });
        sink.on_event(&IndexEvent::RunCompleted {
            processed: 2,
            indexed: 1,
            skipped: 0,
            failed: 1,
            total_chunks: 3,
            elapsed: std::time::Duration::from_secs(1),
            stopped_early: false,
        });

        let snapshot = receiver.borrow().clone();
        assert_eq!(snapshot.processed, 2);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.indexed, 1);
        assert_eq!(snapshot.total, 4);
        assert_eq!(snapshot.total_chunks, 3);
        assert!(snapshot.complete);
        assert!(!snapshot.stopped_early);
    }
}
