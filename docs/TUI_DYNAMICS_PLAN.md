# TUI Dynamics Implementation Plan

## Problem Statement
The TUI currently relies on a fragile heuristic (`infer_embedding_dimension` in `src/embeddings/mod.rs`) that hardcodes model names to embedding dimensions (e.g., `2560` for Qwen 4B, `4096` for 8B). This is factually incorrect because Qwen models often use MRL (Matryoshka Representation Learning), meaning an 8B model could be truncated to 2560 dimensions, or vice-versa. The TUI forces this false heuristic, resulting in confusing prints like `2560 if is_qwen => "Qwen3 Embedding 4B or MRL-truncated 8B (2560 dims)"` in `src/tui/detection.rs`.

## Scope of Changes

### 1. Remove Heuristics (`src/embeddings/mod.rs`)
- Remove the `infer_embedding_dimension` function.
- Remove fallbacks like `DEFAULT_REQUIRED_DIMENSION = 2560` that mask the need for actual dimension probing.

### 2. Enforce Live Probing (`src/tui/app.rs`)
- `EmbedderState` must only trust dimensions obtained via a live probe (`DimensionTruth::Probed`) or explicitly provided by the user (`DimensionTruth::Manual`).
- Remove `DimensionTruth::Inferred` and `DimensionTruth::Default` from the state.
- Do not allow the configuration wizard to proceed or save unless the dimension has been definitively proven (probed) or manually set.

### 3. Clean UI Explanations (`src/tui/detection.rs`)
- Simplify the `dimension_explanation` function. Instead of guessing the model variant based on the dimension, it should simply state the verified dimension.
- Remove the hardcoded text snippets like `"Qwen3 Embedding 4B or MRL-truncated 8B (2560 dims)"`.

### 4. Require Strict Validation
- If a provider is offline, the user must input the dimension manually. The system should no longer "guess" the dimension based on the string name.

## Migration Steps
1. Delete `infer_embedding_dimension` from `src/embeddings/mod.rs`.
2. Update `EmbedderState` in `src/tui/app.rs` to remove the removed `DimensionTruth` variants.
3. Update `apply_detected_provider` and `refresh_manual_dimension_state` in `src/tui/app.rs` to rely exclusively on `schedule_dimension_probe`.
4. Refactor `dimension_explanation` in `src/tui/detection.rs` to format dynamically derived dimension outputs cleanly.
