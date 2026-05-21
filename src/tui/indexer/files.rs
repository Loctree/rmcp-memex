//! File collection helpers for the data setup step.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use walkdir::WalkDir;

/// Supported file extensions for indexing.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "rst", "org", "json", "yaml", "yml", "toml", "xml", "rs", "py", "js",
    "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php", "swift", "kt", "scala",
    "sh", "bash", "zsh", "fish", "sql", "graphql", "html", "css", "scss", "sass", "less", "pdf",
];

/// Collect files from a directory for indexing.
pub fn collect_indexable_files(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if !dir_path.exists() {
        return Err(anyhow!("Directory does not exist: {}", dir_path.display()));
    }

    if !dir_path.is_dir() {
        return Err(anyhow!("Path is not a directory: {}", dir_path.display()));
    }

    for entry in WalkDir::new(dir_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();

        if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }

        if path.is_dir() {
            continue;
        }

        if let Some(ext) = path.extension().and_then(|ext| ext.to_str())
            && SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
        {
            files.push(path.to_path_buf());
        }
    }

    files.sort();
    Ok(files)
}

/// Validate a user-provided path with the repo's path safety rules.
pub fn validate_path(path: &str) -> Result<PathBuf> {
    use crate::path_utils::sanitize_existing_path;

    if path.trim().is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    sanitize_existing_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_path_rejects_empty_values() {
        assert!(validate_path("").is_err());
        assert!(validate_path("   ").is_err());
    }
}
