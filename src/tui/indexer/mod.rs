//! Indexer module for the TUI wizard.

mod contracts;
mod files;
mod import;
mod scheduler;
mod sinks;
mod state;

pub use contracts::{
    INDEX_CONTROL_CHANNEL_CAPACITY, IndexControl, IndexEvent, IndexEventSink,
    IndexTelemetrySnapshot, SharedIndexTelemetry, new_index_telemetry,
};
pub use files::{collect_indexable_files, validate_path};
pub use import::import_lancedb;
pub use scheduler::start_indexing;
pub use sinks::{FanOut, TracingSink, TuiTelemetrySink};
pub use state::{DataSetupOption, DataSetupState, DataSetupSubStep, ImportMode};
