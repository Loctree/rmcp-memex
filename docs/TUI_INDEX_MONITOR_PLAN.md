# TUI Index Monitor Plan

## Current State

The existing TUI wizard already has a solid shell, but the indexing step is still too thin for real operations.

- The indexing screen shows only basic progress.
- There is no live CPU/GPU/RAM telemetry.
- There is no runtime control over indexing speed.
- The TUI indexer path is effectively single-gear from the operator's point of view.

Relevant code:

- `src/tui/app.rs`
- `src/tui/ui.rs`
- `src/tui/indexer.rs`

## Target Shape

Turn the `Data Setup -> Indexing` step into a real operator dashboard inside the existing Crossterm/Ratatui TUI.

It should provide:

- live system monitor: CPU, RAM, `rmcp-memex`, embedder processes, GPU
- live indexing monitor: files processed, rate, ETA, skipped, failed, inflight, current file
- runtime controls: pause/resume and parallelism up/down
- honest telemetry on macOS, with best-effort GPU metrics via `ioreg`

## Architecture Plan

### 1. Extend indexing progress model

Expand `IndexProgress` in `src/tui/indexer.rs` so it carries richer runtime state:

- `processed`
- `total`
- `skipped`
- `failed`
- `inflight`
- `parallelism`
- `paused`
- `files_per_sec`
- `eta_seconds`
- `elapsed_seconds`
- `current_file`
- `complete`
- `error`

This gives the UI one truthful state packet instead of forcing it to infer behavior.

### 2. Add index control channel

Add a control path from the TUI into the indexing task.

Suggested command enum:

```rust
enum IndexControl {
    Pause,
    Resume,
    SetParallelism(usize),
    Stop,
}
```

This will let the operator tune the job without restarting it.

### 3. Rework TUI indexing runtime into a scheduler

Replace the simple sequential loop in `start_indexing()` / `run_indexing()` with a bounded scheduler that:

- tracks a work queue
- controls concurrent file tasks
- reacts to `IndexControl`
- updates progress snapshots continuously

This is the core enabling step for the regulator.

### 4. Add telemetry module

Create a new module:

- `src/tui/monitor.rs`

It should sample:

- system CPU load
- system memory usage / pressure proxy
- `rmcp-memex` process CPU and RSS
- embedder process aggregate CPU and RSS
- GPU utilization and memory on macOS via `ioreg`

The GPU path should be best-effort and honest:

- use `IOAccelerator` / `AGXAccelerator`
- parse `Device Utilization %`
- parse `In use system memory`
- fall back gracefully when metrics are unavailable

### 5. Integrate monitor state into App

Extend `App` in `src/tui/app.rs` with:

- monitor state snapshot
- receiver for telemetry updates
- sender for index controls

This should mirror the current `index_progress_rx` pattern so the event loop stays simple.

### 6. Upgrade indexing screen into dashboard

Rebuild the `DataSetupSubStep::Indexing` view in `src/tui/ui.rs` into a real dashboard.

Recommended layout:

- left pane: progress gauge, rate, ETA, current file, pause state, parallelism
- right pane: system CPU/RAM, `rmcp-memex`, embedder aggregate, GPU
- bottom strip: controls and recent status notes

Use gauges and compact stat blocks rather than a long wall of text.

### 7. Add keybindings

During active indexing:

- `Space` -> pause/resume
- `+` / `=` -> increase parallelism
- `-` -> decrease parallelism
- `s` -> stop after current inflight batch or request shutdown

Footer help text should reflect the active control mode.

### 8. Verification

Before merging:

- `cargo test`
- `cargo clippy -- -D warnings`

Also verify manually in a real index run that:

- progress updates stay smooth
- pause/resume works
- parallelism changes are visible in runtime behavior
- GPU fallback does not break on unsupported machines

## Delivery Strategy

### Phase 1

Ship the monitor first:

- richer progress state
- telemetry sampler
- dashboard UI

### Phase 2

Ship the regulator:

- pause/resume
- live parallelism changes
- stop control

This split gets visible value quickly without mixing UI polish and concurrency surgery into one blind jump.

## Quick Win

The smallest sharp move is:

1. add `monitor.rs`
2. show CPU/RAM/GPU + rate/ETA on the indexing screen
3. keep control static for the first pass

That already turns the TUI from a setup wizard into a useful operator console.
