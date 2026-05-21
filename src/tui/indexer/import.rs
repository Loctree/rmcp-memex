//! LanceDB import helpers for the data setup step.

use std::path::Path;

use anyhow::{Result, anyhow};

use super::state::ImportMode;

/// Import an existing LanceDB database.
pub async fn import_lancedb(
    source_path: &Path,
    target_path: &Path,
    mode: ImportMode,
) -> Result<String> {
    let source = source_path.to_path_buf();

    if !source.exists() {
        return Err(anyhow!("Source path does not exist: {}", source.display()));
    }

    match mode {
        ImportMode::Copy => {
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            copy_dir_recursive(&source, target_path).await?;
            Ok(format!("Copied database to {}", target_path.display()))
        }
        ImportMode::Symlink => {
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            if target_path.exists() || target_path.is_symlink() {
                if target_path.is_dir() {
                    tokio::fs::remove_dir_all(target_path).await?;
                } else {
                    tokio::fs::remove_file(target_path).await?;
                }
            }

            #[cfg(unix)]
            tokio::fs::symlink(&source, target_path).await?;
            #[cfg(windows)]
            tokio::fs::symlink_dir(&source, target_path).await?;

            Ok(format!(
                "Created symlink {} -> {}",
                target_path.display(),
                source.display()
            ))
        }
        ImportMode::ConfigOnly => Ok(format!(
            "Config will use existing database at {}",
            source.display()
        )),
    }
}

/// Recursively copy a directory with path validation.
pub async fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    use crate::path_utils::validate_write_path;

    let (_safe_src, mut entries) = crate::path_utils::safe_read_dir(src).await?;
    let safe_dst = validate_write_path(dst)?;

    tokio::fs::create_dir_all(&safe_dst).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let entry_name = entry.file_name();
        let entry_name_str = entry_name.to_string_lossy();

        if entry_name_str.contains("..")
            || (entry_name_str.starts_with('.')
                && entry_name_str.len() > 1
                && entry_name_str.chars().nth(1) == Some('.'))
        {
            continue;
        }

        let destination = safe_dst.join(&entry_name);

        if path.is_dir() {
            Box::pin(copy_dir_recursive(&path, &destination)).await?;
        } else if let Ok(_copied) = crate::path_utils::safe_copy(&path, &destination) {
            // File copied successfully.
        }
    }

    Ok(())
}
