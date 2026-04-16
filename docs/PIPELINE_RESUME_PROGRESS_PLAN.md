# Pipeline Resume + Progress Plan

## Status Note

This document started as a forward plan. The core work is now implemented:

- `--pipeline` supports `--progress`
- `--pipeline` supports `--resume`
- adaptive pipeline governor controls batch size, batch chars, and embed concurrency
- pipeline progress snapshots stream live queue depths, rates, ETA, and governor state

What remained worth tightening after the original implementation was operator truth
during resumed runs: live progress now carries discovered-file and resumed-file counts
so the runtime stream reflects the whole run, not only the remaining scheduled subset.

## Current State

The async pipeline path already has the right broad shape:

- reader
- chunker
- embedder
- storage

It overlaps I/O, chunk creation, embedding, and writes through bounded channels in `src/rag/pipeline.rs`.

The remaining work from the original plan is mostly future refinement, not baseline
capability recovery.

Relevant code:

- `src/rag/pipeline.rs`
- `src/bin/cli/maintenance.rs`
- `src/embeddings/mod.rs`

## Better Shape

Pipeline mode should become the canonical high-throughput path, not the "fast but blind" path.

It should support:

- live progress and ETA
- truthful resume after interruption
- adaptive throughput control based on observed embedder pressure
- preservation of current backpressure architecture

## Design Principles

### 1. Resume must be commit-based, not read-based

Do not checkpoint:

- file discovered
- file read
- chunks created
- chunks embedded

Checkpoint only when a file is durably committed to storage.

That is the only moment where resume becomes truthful.

### 2. Progress must come from stage facts, not guesses

Expose runtime snapshots based on real counters from each stage:

- files discovered
- files read
- files skipped
- files committed
- chunks created
- chunks embedded
- chunks stored
- stage-local errors
- queue depth between stages
- rolling rate / ETA

### 3. GPU control should be flow control, not fake hardware control

Do not pretend we are scheduling Apple GPU cores directly.

What we can truthfully control:

- batch size
- max chars per batch
- number of concurrent embed requests
- channel depths / backpressure thresholds

That is enough to "govern GPU usage" in practice.

## Architecture Plan

### 1. Add pipeline progress events

Introduce a progress/event stream emitted from the pipeline coordinator and stages.

Suggested shape:

```rust
enum PipelineEvent {
    FileRead { path: PathBuf },
    FileSkipped { path: PathBuf, reason: String },
    ChunksCreated { path: PathBuf, count: usize },
    ChunksEmbedded { path: PathBuf, count: usize },
    FileCommitted { path: PathBuf, chunk_count: usize },
    Error { path: Option<PathBuf>, stage: &'static str, message: String },
    Snapshot(PipelineSnapshot),
}
```

This keeps CLI and future TUI consumers decoupled from implementation details.

### 2. Add stage-aware snapshot model

Define a `PipelineSnapshot` that carries runtime state:

- total files
- files read
- files skipped
- files committed
- chunks created
- chunks embedded
- chunks stored
- queue depths
- current embed batch size
- rolling files/sec
- rolling chunks/sec
- ETA
- elapsed
- errors

CLI `--progress` should render from this stream rather than only printing final totals.

### 3. Preserve file boundaries through the pipeline

Right now storage sees only batches of `EmbeddedChunk`.

To make resume truthful, storage needs to know when all chunks for a given source file have been flushed successfully.

Refactor the payload shape so file identity survives until commit time.

For example:

- wrap per-file chunks as a unit through the pipeline
- or annotate each embedded chunk batch with file completion markers

The target is simple:

- storage stage can emit `FileCommitted(path)`

### 4. Add checkpoint support for pipeline mode

Reuse the existing checkpoint concept from CLI maintenance, but move semantics to commit-based tracking.

Checkpoint contents should include:

- namespace
- db path
- committed file set
- last updated timestamp
- optional stats snapshot

On startup with `--resume`:

- load checkpoint
- filter already committed files before reader stage
- continue only with unfinished files

On successful full completion:

- remove checkpoint

On partial failure:

- preserve checkpoint

### 5. Add progress rendering to CLI pipeline mode

Replace today's warnings in `src/bin/cli/maintenance.rs` with real support:

- interactive progress bar when terminal is interactive
- line-based snapshots when non-interactive

The operator should see:

- current file count
- chunk flow
- stage bottleneck
- effective embedding rate
- ETA

### 6. Add adaptive throughput governor

Add an optional governor that tunes embed workload based on runtime signals.

Candidate inputs:

- rolling embed latency
- batch failure / retry rate
- queue depth before embedder
- queue depth before storage
- macOS GPU utilization from `ioreg`
- GPU memory / driver memory from `IOAccelerator`

Candidate outputs:

- lower / raise `max_batch_items`
- lower / raise `max_batch_chars`
- lower / raise concurrent embed requests

This should be conservative and monotonic:

- increase slowly
- decrease quickly on pressure

### 7. Keep manual override stronger than automation

Adaptive mode should be optional.

Operators should still be able to force:

- fixed batching
- fixed concurrency
- governor off

The system should help, not seize control.

## Implementation Order

### Phase 1: Visibility

- add `PipelineEvent`
- add `PipelineSnapshot`
- wire live progress to CLI

### Phase 2: Truthful resume

- preserve file boundaries to storage
- emit `FileCommitted`
- add checkpointing for pipeline mode

### Phase 3: Adaptive governor

- add runtime telemetry inputs
- adapt batching / concurrency
- expose current governor decisions in progress output

## Risks

### 1. False resume semantics

If checkpointing happens before durable storage completion, resume will lie and can silently skip unfinished work.

### 2. Memory blow-up

If file boundaries are preserved naively, large per-file buffers can explode RAM.

Mitigation:

- preserve file identity without forcing entire file payloads to stay buffered in memory longer than needed

### 3. Overactive governor

If adaptive control changes knobs too aggressively, throughput will oscillate.

Mitigation:

- rolling averages
- hysteresis
- clamp ranges
- cool-down intervals

## Quick Win

The smallest sharp move is:

1. add live progress to pipeline mode
2. expose queue depths and embed rate
3. stop printing "progress unsupported"

That alone turns pipeline mode from a black box into an operational path worth trusting.
