//! Concurrent indexer scheduler with pause/resume/stop controls.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use chrono::Utc;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt, stream::FuturesUnordered};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError, mpsc};
use tokio::task::JoinHandle;

use crate::{
    EmbeddingClient, EmbeddingConfig, IndexResult, RAGPipeline, SliceMode, StorageManager,
};

use super::contracts::{IndexControl, IndexEvent, IndexEventSink};

type FileProcessor =
    Arc<dyn Fn(usize, PathBuf, String) -> BoxFuture<'static, FileOutcome> + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum FileOutcome {
    Indexed {
        file_index: usize,
        path: PathBuf,
        chunks_indexed: usize,
        content_hash: String,
        duration_ms: u64,
        embedder_ms: Option<u64>,
        tokens_estimated: Option<usize>,
    },
    Skipped {
        file_index: usize,
        path: PathBuf,
        reason: String,
        content_hash: Option<String>,
    },
    Failed {
        file_index: usize,
        path: PathBuf,
        error: String,
    },
}

struct SchedulerState {
    pending: VecDeque<(usize, PathBuf)>,
    inflight: FuturesUnordered<JoinHandle<FileOutcome>>,
    semaphore: Arc<Semaphore>,
    resume_notify: Arc<Notify>,
    namespace: String,
    parallelism: usize,
    paused: bool,
    stop_requested: bool,
    indexed: usize,
    skipped: usize,
    failed: usize,
    total_chunks: usize,
    started_at: Instant,
    total: usize,
}

impl SchedulerState {
    fn processed(&self) -> usize {
        self.indexed + self.skipped + self.failed
    }

    fn in_flight(&self) -> usize {
        self.inflight.len()
    }

    fn files_per_sec(&self) -> f64 {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            self.processed() as f64 / elapsed
        }
    }

    fn eta_secs(&self) -> Option<f64> {
        let rate = self.files_per_sec();
        if rate <= f64::EPSILON {
            None
        } else {
            Some(self.total.saturating_sub(self.processed()) as f64 / rate)
        }
    }
}

/// Start the concurrent indexing scheduler.
#[allow(clippy::too_many_arguments)]
pub fn start_indexing(
    source_dir: PathBuf,
    files: Vec<PathBuf>,
    namespace: String,
    embedding_config: EmbeddingConfig,
    db_path: String,
    sink: Arc<dyn IndexEventSink>,
    control_rx: mpsc::Receiver<IndexControl>,
    initial_parallelism: usize,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let expanded_db_path = shellexpand::tilde(&db_path).to_string();
        let storage = Arc::new(StorageManager::new_lance_only(&expanded_db_path).await?);
        storage.ensure_collection().await?;

        let embedding_client = Arc::new(tokio::sync::Mutex::new(
            EmbeddingClient::new(&embedding_config).await?,
        ));
        let pipeline = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

        let processor: FileProcessor = Arc::new(move |file_index, path, namespace| {
            let pipeline = pipeline.clone();
            async move {
                let started_at = Instant::now();
                match pipeline
                    .index_document_with_dedup(&path, Some(&namespace), SliceMode::Onion)
                    .await
                {
                    Ok(IndexResult::Indexed {
                        chunks_indexed,
                        content_hash,
                        embedder_ms,
                        tokens_estimated,
                    }) => FileOutcome::Indexed {
                        file_index,
                        path,
                        chunks_indexed,
                        content_hash,
                        duration_ms: started_at.elapsed().as_millis() as u64,
                        embedder_ms,
                        tokens_estimated,
                    },
                    Ok(IndexResult::Skipped {
                        reason,
                        content_hash,
                    }) => FileOutcome::Skipped {
                        file_index,
                        path,
                        reason,
                        content_hash: Some(content_hash),
                    },
                    Err(error) => FileOutcome::Failed {
                        file_index,
                        path,
                        error: error.to_string(),
                    },
                }
            }
            .boxed()
        });

        run_scheduler_with_processor(
            source_dir,
            files,
            namespace,
            sink,
            control_rx,
            initial_parallelism,
            processor,
        )
        .await
    })
}

async fn run_scheduler_with_processor(
    source_dir: PathBuf,
    files: Vec<PathBuf>,
    namespace: String,
    sink: Arc<dyn IndexEventSink>,
    mut control_rx: mpsc::Receiver<IndexControl>,
    initial_parallelism: usize,
    processor: FileProcessor,
) -> Result<()> {
    let parallelism = initial_parallelism.max(1);
    let mut state = SchedulerState {
        total: files.len(),
        pending: files.into_iter().enumerate().collect(),
        inflight: FuturesUnordered::new(),
        semaphore: Arc::new(Semaphore::new(parallelism)),
        resume_notify: Arc::new(Notify::new()),
        namespace,
        parallelism,
        paused: false,
        stop_requested: false,
        indexed: 0,
        skipped: 0,
        failed: 0,
        total_chunks: 0,
        started_at: Instant::now(),
    };

    sink.on_event(&IndexEvent::RunStarted {
        total_files: state.total,
        namespace: state.namespace.clone(),
        source_dir: source_dir.display().to_string(),
        parallelism: state.parallelism,
        started_at: Utc::now(),
    });
    emit_stats_tick(&state, &sink);

    let mut stats_interval = tokio::time::interval(tokio::time::Duration::from_millis(500));

    loop {
        drain_control_queue(&mut state, &sink, &mut control_rx);

        if state.stop_requested {
            if state.inflight.is_empty() {
                break;
            }

            tokio::select! {
                _ = stats_interval.tick() => {
                    emit_stats_tick(&state, &sink);
                }
                Some(control) = control_rx.recv() => {
                    handle_control(&mut state, &sink, control);
                }
                Some(join_result) = state.inflight.next() => {
                    apply_join_result(&mut state, &sink, join_result)?;
                }
            }
            continue;
        }

        if state.paused {
            let notify = state.resume_notify.clone();
            let resume_wait = notify.notified();
            tokio::pin!(resume_wait);

            tokio::select! {
                _ = stats_interval.tick() => {
                    emit_stats_tick(&state, &sink);
                }
                Some(control) = control_rx.recv() => {
                    handle_control(&mut state, &sink, control);
                }
                Some(join_result) = state.inflight.next(), if !state.inflight.is_empty() => {
                    apply_join_result(&mut state, &sink, join_result)?;
                }
                _ = &mut resume_wait => {}
            }
            continue;
        }

        spawn_ready_tasks(&mut state, &sink, processor.clone());

        if state.pending.is_empty() && state.inflight.is_empty() {
            break;
        }

        tokio::select! {
            _ = stats_interval.tick() => {
                emit_stats_tick(&state, &sink);
            }
            Some(control) = control_rx.recv() => {
                handle_control(&mut state, &sink, control);
            }
            Some(join_result) = state.inflight.next(), if !state.inflight.is_empty() => {
                apply_join_result(&mut state, &sink, join_result)?;
            }
            else => {
                tokio::task::yield_now().await;
            }
        }
    }

    sink.on_event(&IndexEvent::RunCompleted {
        processed: state.processed(),
        indexed: state.indexed,
        skipped: state.skipped,
        failed: state.failed,
        total_chunks: state.total_chunks,
        elapsed: state.started_at.elapsed(),
        stopped_early: state.stop_requested,
    });

    Ok(())
}

fn drain_control_queue(
    state: &mut SchedulerState,
    sink: &Arc<dyn IndexEventSink>,
    control_rx: &mut mpsc::Receiver<IndexControl>,
) {
    while let Ok(control) = control_rx.try_recv() {
        handle_control(state, sink, control);
    }
}

fn spawn_ready_tasks(
    state: &mut SchedulerState,
    sink: &Arc<dyn IndexEventSink>,
    processor: FileProcessor,
) {
    if state.paused || state.stop_requested {
        return;
    }

    while state.in_flight() < state.parallelism && !state.pending.is_empty() {
        let permit = match try_acquire_permit(&state.semaphore) {
            Ok(Some(permit)) => permit,
            Ok(None) => break,
            Err(_) => break,
        };

        let Some((file_index, path)) = state.pending.pop_front() else {
            break;
        };
        let size_bytes = std::fs::metadata(&path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        sink.on_event(&IndexEvent::FileStarted {
            file_index,
            path: path.display().to_string(),
            size_bytes,
        });

        let work = processor.clone();
        let namespace = state.namespace.clone();
        let join_handle = tokio::spawn(async move {
            let _permit = permit;
            work(file_index, path, namespace).await
        });
        state.inflight.push(join_handle);
        emit_stats_tick(state, sink);
    }
}

fn try_acquire_permit(
    semaphore: &Arc<Semaphore>,
) -> Result<Option<OwnedSemaphorePermit>, TryAcquireError> {
    semaphore
        .clone()
        .try_acquire_owned()
        .map(Some)
        .or_else(|error| {
            if matches!(error, TryAcquireError::NoPermits) {
                Ok(None)
            } else {
                Err(error)
            }
        })
}

fn handle_control(
    state: &mut SchedulerState,
    sink: &Arc<dyn IndexEventSink>,
    control: IndexControl,
) {
    match control {
        IndexControl::Pause => {
            if !state.paused && !state.stop_requested {
                state.paused = true;
                sink.on_event(&IndexEvent::Paused);
                emit_stats_tick(state, sink);
            }
        }
        IndexControl::Resume => {
            if state.paused && !state.stop_requested {
                state.paused = false;
                state.resume_notify.notify_waiters();
                sink.on_event(&IndexEvent::Resumed);
                emit_stats_tick(state, sink);
            }
        }
        IndexControl::SetParallelism(level) => {
            let next = level.max(1);
            let previous = state.parallelism;
            if next != previous {
                adjust_parallelism(&state.semaphore, previous, next);
                state.parallelism = next;
                sink.on_event(&IndexEvent::ParallelismChanged {
                    previous,
                    current: next,
                });
                emit_stats_tick(state, sink);
            }
        }
        IndexControl::Stop => {
            if !state.stop_requested {
                state.stop_requested = true;
                state.paused = false;
                state.resume_notify.notify_waiters();
                sink.on_event(&IndexEvent::StopRequested);
                emit_stats_tick(state, sink);
            }
        }
    }
}

fn adjust_parallelism(semaphore: &Arc<Semaphore>, previous: usize, next: usize) {
    if next > previous {
        semaphore.add_permits(next - previous);
        return;
    }

    for _ in 0..(previous - next) {
        match semaphore.try_acquire() {
            Ok(permit) => permit.forget(),
            Err(TryAcquireError::NoPermits) | Err(TryAcquireError::Closed) => break,
        }
    }
}

fn apply_join_result(
    state: &mut SchedulerState,
    sink: &Arc<dyn IndexEventSink>,
    join_result: Result<FileOutcome, tokio::task::JoinError>,
) -> Result<()> {
    let outcome = match join_result {
        Ok(outcome) => outcome,
        Err(error) => {
            let message = format!("indexing task join failed: {error}");
            sink.on_event(&IndexEvent::RunFailed {
                error: message.clone(),
                processed_before_failure: state.processed(),
            });
            return Err(anyhow!(message));
        }
    };

    apply_outcome(state, sink, outcome);
    Ok(())
}

fn apply_outcome(state: &mut SchedulerState, sink: &Arc<dyn IndexEventSink>, outcome: FileOutcome) {
    match outcome {
        FileOutcome::Indexed {
            file_index,
            path,
            chunks_indexed,
            content_hash,
            duration_ms,
            embedder_ms,
            tokens_estimated,
        } => {
            state.indexed += 1;
            state.total_chunks += chunks_indexed;
            sink.on_event(&IndexEvent::FileIndexed {
                file_index,
                path: path.display().to_string(),
                chunks_indexed,
                content_hash,
                duration_ms,
                embedder_ms,
                tokens_estimated,
            });
        }
        FileOutcome::Skipped {
            file_index,
            path,
            reason,
            content_hash,
        } => {
            state.skipped += 1;
            sink.on_event(&IndexEvent::FileSkipped {
                file_index,
                path: path.display().to_string(),
                reason,
                content_hash,
            });
        }
        FileOutcome::Failed {
            file_index,
            path,
            error,
        } => {
            state.failed += 1;
            sink.on_event(&IndexEvent::FileFailed {
                file_index,
                path: path.display().to_string(),
                error,
            });
        }
    }

    emit_stats_tick(state, sink);
}

fn emit_stats_tick(state: &SchedulerState, sink: &Arc<dyn IndexEventSink>) {
    sink.on_event(&IndexEvent::StatsTick {
        processed: state.processed(),
        indexed: state.indexed,
        skipped: state.skipped,
        failed: state.failed,
        total: state.total,
        files_per_sec: state.files_per_sec(),
        eta_secs: state.eta_secs(),
        total_chunks: state.total_chunks,
        in_flight: state.in_flight(),
    });
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    use super::*;
    use crate::tui::indexer::contracts::INDEX_CONTROL_CHANNEL_CAPACITY;

    struct RecordingSink {
        events: Arc<StdMutex<Vec<IndexEvent>>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                events: Arc::new(StdMutex::new(Vec::new())),
            }
        }

        fn events(&self) -> Vec<IndexEvent> {
            self.events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    impl IndexEventSink for RecordingSink {
        fn on_event(&self, event: &IndexEvent) {
            self.events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(event.clone());
        }
    }

    fn test_files(count: usize) -> Vec<PathBuf> {
        (0..count)
            .map(|index| Path::new("/tmp").join(format!("file-{index}.txt")))
            .collect()
    }

    #[tokio::test]
    async fn scheduler_pause_resume_blocks_new_starts_until_resumed() {
        let sink = Arc::new(RecordingSink::new());
        let (control_tx, control_rx) = mpsc::channel(INDEX_CONTROL_CHANNEL_CAPACITY);

        let processor: FileProcessor = Arc::new(move |file_index, path, _namespace| {
            async move {
                tokio::time::sleep(Duration::from_millis(80)).await;
                FileOutcome::Indexed {
                    file_index,
                    path,
                    chunks_indexed: 1,
                    content_hash: format!("hash-{file_index}"),
                    duration_ms: 5,
                    embedder_ms: Some(5),
                    tokens_estimated: Some(10),
                }
            }
            .boxed()
        });

        let join = tokio::spawn(run_scheduler_with_processor(
            PathBuf::from("/tmp"),
            test_files(10),
            "kb:test".to_string(),
            sink.clone(),
            control_rx,
            2,
            processor,
        ));

        tokio::time::sleep(Duration::from_millis(30)).await;
        control_tx
            .send(IndexControl::Pause)
            .await
            .expect("send pause");
        tokio::time::sleep(Duration::from_millis(30)).await;

        let events_after_pause = sink.events();
        let started_before_resume = events_after_pause
            .iter()
            .filter(|event| matches!(event, IndexEvent::FileStarted { .. }))
            .count();
        assert!(
            events_after_pause
                .iter()
                .any(|event| matches!(event, IndexEvent::Paused))
        );

        tokio::time::sleep(Duration::from_millis(60)).await;
        let events_still_paused = sink.events();
        let started_while_paused = events_still_paused
            .iter()
            .filter(|event| matches!(event, IndexEvent::FileStarted { .. }))
            .count();
        assert_eq!(started_while_paused, started_before_resume);

        control_tx
            .send(IndexControl::Resume)
            .await
            .expect("send resume");

        join.await
            .expect("scheduler join")
            .expect("scheduler result");

        let final_events = sink.events();
        assert!(
            final_events
                .iter()
                .any(|event| matches!(event, IndexEvent::Resumed))
        );
        let final_started = final_events
            .iter()
            .filter(|event| matches!(event, IndexEvent::FileStarted { .. }))
            .count();
        assert_eq!(final_started, 10);
    }

    #[tokio::test]
    async fn scheduler_stop_drains_inflight_and_completes_cleanly() {
        let sink = Arc::new(RecordingSink::new());
        let (control_tx, control_rx) = mpsc::channel(INDEX_CONTROL_CHANNEL_CAPACITY);

        let processor: FileProcessor = Arc::new(move |file_index, path, _namespace| {
            async move {
                tokio::time::sleep(Duration::from_millis(80)).await;
                FileOutcome::Indexed {
                    file_index,
                    path,
                    chunks_indexed: 1,
                    content_hash: format!("hash-{file_index}"),
                    duration_ms: 5,
                    embedder_ms: None,
                    tokens_estimated: None,
                }
            }
            .boxed()
        });

        let join = tokio::spawn(run_scheduler_with_processor(
            PathBuf::from("/tmp"),
            test_files(100),
            "kb:test".to_string(),
            sink.clone(),
            control_rx,
            4,
            processor,
        ));

        tokio::time::sleep(Duration::from_millis(30)).await;
        control_tx
            .send(IndexControl::Stop)
            .await
            .expect("send stop");

        join.await
            .expect("scheduler join")
            .expect("scheduler result");

        let events = sink.events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, IndexEvent::StopRequested))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, IndexEvent::RunCompleted { .. }))
        );
    }
}
