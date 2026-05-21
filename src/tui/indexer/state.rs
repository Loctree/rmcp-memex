//! Wizard state for the data setup step.

use std::path::PathBuf;

/// Data setup option selected by user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSetupOption {
    /// Import an existing LanceDB database.
    ImportLanceDB,
    /// Index a directory with files.
    IndexDirectory,
    /// Skip data setup for now.
    Skip,
}

impl DataSetupOption {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ImportLanceDB => "[1] Import existing LanceDB",
            Self::IndexDirectory => "[2] Index a directory now",
            Self::Skip => "[3] Skip for now",
        }
    }

    pub fn detail(&self) -> &'static str {
        match self {
            Self::ImportLanceDB => "Copy or link an existing LanceDB database",
            Self::IndexDirectory => "Recursively index files with embeddings",
            Self::Skip => "Configure data later via CLI",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![Self::ImportLanceDB, Self::IndexDirectory, Self::Skip]
    }
}

/// State for the data setup wizard step.
#[derive(Debug, Clone)]
pub struct DataSetupState {
    pub option: DataSetupOption,
    pub focus: usize,
    pub input_mode: bool,
    pub input_buffer: String,
    pub source_path: Option<String>,
    pub namespace: Option<String>,
    pub sub_step: DataSetupSubStep,
    pub import_mode: ImportMode,
    pub validation_error: Option<String>,
}

/// Sub-steps within data setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSetupSubStep {
    SelectOption,
    EnterPath,
    EnterNamespace,
    SelectImportMode,
    Indexing,
    Complete,
}

/// Mode for importing LanceDB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportMode {
    Copy,
    Symlink,
    ConfigOnly,
}

impl ImportMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Copy => "[1] Copy database files",
            Self::Symlink => "[2] Create symlink",
            Self::ConfigOnly => "[3] Just update config path",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![Self::Copy, Self::Symlink, Self::ConfigOnly]
    }
}

impl Default for DataSetupState {
    fn default() -> Self {
        Self {
            option: DataSetupOption::Skip,
            focus: 0,
            input_mode: false,
            input_buffer: String::new(),
            source_path: None,
            namespace: None,
            sub_step: DataSetupSubStep::SelectOption,
            import_mode: ImportMode::ConfigOnly,
            validation_error: None,
        }
    }
}

impl DataSetupState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn focused_option(&self) -> DataSetupOption {
        let options = DataSetupOption::all();
        options
            .get(self.focus)
            .cloned()
            .unwrap_or(DataSetupOption::Skip)
    }

    pub fn select_focused(&mut self) {
        self.option = self.focused_option();
        self.validation_error = None;
        self.sub_step = match self.option {
            DataSetupOption::ImportLanceDB | DataSetupOption::IndexDirectory => {
                DataSetupSubStep::EnterPath
            }
            DataSetupOption::Skip => DataSetupSubStep::Complete,
        };
        if self.sub_step == DataSetupSubStep::EnterPath {
            self.input_mode = true;
            self.input_buffer.clear();
        }
    }

    pub fn confirm_path(&mut self) {
        let path = self.input_buffer.trim().to_string();
        if path.is_empty() {
            return;
        }
        self.validation_error = None;
        self.source_path = Some(path);
        self.input_mode = false;

        match self.option {
            DataSetupOption::ImportLanceDB => {
                self.sub_step = DataSetupSubStep::SelectImportMode;
                self.focus = 0;
            }
            DataSetupOption::IndexDirectory => {
                if let Some(ref source_path) = self.source_path {
                    let folder_name = PathBuf::from(source_path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("indexed")
                        .to_string();
                    self.input_buffer = format!("kb:{folder_name}");
                }
                self.sub_step = DataSetupSubStep::EnterNamespace;
                self.input_mode = true;
            }
            DataSetupOption::Skip => {
                self.sub_step = DataSetupSubStep::Complete;
            }
        }
    }

    pub fn confirm_namespace(&mut self) {
        let namespace = self.input_buffer.trim();
        self.validation_error = None;
        self.namespace = Some(if namespace.is_empty() {
            "rag".to_string()
        } else {
            namespace.to_string()
        });
        self.input_mode = false;
        self.sub_step = DataSetupSubStep::Indexing;
    }

    pub fn select_import_mode(&mut self, mode: ImportMode) {
        self.import_mode = mode;
        self.sub_step = DataSetupSubStep::Complete;
    }

    pub fn is_done(&self) -> bool {
        self.sub_step == DataSetupSubStep::Complete
    }

    pub fn is_indexing(&self) -> bool {
        self.sub_step == DataSetupSubStep::Indexing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_transitions_select_directory_flow() {
        let mut state = DataSetupState::new();
        state.focus = 1;
        state.select_focused();
        assert_eq!(state.option, DataSetupOption::IndexDirectory);
        assert_eq!(state.sub_step, DataSetupSubStep::EnterPath);

        state.input_buffer = "/tmp/docs".to_string();
        state.confirm_path();
        assert_eq!(state.sub_step, DataSetupSubStep::EnterNamespace);

        state.input_buffer = "kb:docs".to_string();
        state.confirm_namespace();
        assert_eq!(state.namespace.as_deref(), Some("kb:docs"));
        assert!(state.is_indexing());
    }
}
