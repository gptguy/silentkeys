use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tauri::AppHandle;

use crate::asr::recognizer::AsrError;

use super::{download, MODEL_BASE_URL, MODEL_FILES};

fn missing_model_files(snapshot_dir: &Path) -> Vec<String> {
    MODEL_FILES
        .iter()
        .filter_map(|file| {
            let path = snapshot_dir.join(file);
            if path.exists() {
                None
            } else {
                Some((*file).to_string())
            }
        })
        .collect()
}

pub fn missing_model_files_for_tests(snapshot_dir: &Path) -> Vec<String> {
    missing_model_files(snapshot_dir)
}

pub fn default_model_root(app: &AppHandle) -> PathBuf {
    if let Some(path) = crate::settings::get_custom_model_path(app) {
        return path;
    }

    fallback_model_root()
}

pub fn fallback_model_root() -> PathBuf {
    let base = dirs_next::cache_dir()
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));

    base.join("huggingface")
        .join("hub")
        .join("models--istupakov--parakeet-tdt-0.6b-v3-onnx")
}

fn ensure_snapshot_complete(root: &Path, snapshot_dir: PathBuf) -> Result<PathBuf, AsrError> {
    if !snapshot_dir.is_dir() {
        log::warn!(
            "Model snapshot directory missing at {}. Downloading fresh snapshot.",
            snapshot_dir.display()
        );
        return download::download_default_snapshot(root);
    }

    let missing = missing_model_files(&snapshot_dir);
    if missing.is_empty() {
        return Ok(snapshot_dir);
    }

    log::warn!(
        "Model snapshot at {} missing required files ({}). Attempting to download missing assets.",
        snapshot_dir.display(),
        missing.join(", ")
    );

    match download::download_missing_files(&snapshot_dir, &missing) {
        Ok(()) => Ok(snapshot_dir),
        Err(err) => {
            log::warn!(
                "Snapshot repair failed, falling back to fresh download: {}",
                err
            );
            download::download_default_snapshot(root)
        }
    }
}

pub fn resolve_model_dir<P: AsRef<Path>>(root: P) -> Result<PathBuf, AsrError> {
    let root = root.as_ref();
    log::debug!("resolve_model_dir: checking root {}", root.display());

    let refs_main = root.join("refs").join("main");

    if refs_main.exists() {
        let commit = fs::read_to_string(&refs_main)?.trim().to_string();
        let snap = root.join("snapshots").join(&commit);
        log::debug!(
            "Found refs/main: {}, checking snapshot at {}",
            commit,
            snap.display()
        );
        if snap.is_dir() {
            log::debug!("Snapshot directory exists and is valid.");
            return ensure_snapshot_complete(root, snap);
        } else {
            log::warn!("Snapshot directory from refs/main does NOT exist or is not a dir.");
        }
    } else {
        log::debug!("refs/main not found at {}", refs_main.display());
    }

    let snapshots = root.join("snapshots");
    if snapshots.is_dir() {
        log::debug!("Scanning snapshots dir: {}", snapshots.display());
        let mut newest: Option<(SystemTime, PathBuf)> = None;
        for entry in fs::read_dir(&snapshots)? {
            let entry = entry?;
            let path = entry.path();

            if !entry.file_type()?.is_dir() {
                log::trace!("Skipping non-dir: {}", path.display());
                continue;
            }

            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);

            match &mut newest {
                Some((ts, best)) if modified > *ts => {
                    log::trace!(
                        "Newer snapshot found: {} (ts={:?})",
                        path.display(),
                        modified
                    );
                    *ts = modified;
                    *best = path;
                }
                None => newest = Some((modified, path)),
                _ => {}
            }
        }

        if let Some((_, path)) = newest {
            log::info!("Selected newest snapshot: {}", path.display());
            return ensure_snapshot_complete(root, path);
        } else {
            log::warn!("No valid directories found in snapshots folder.");
        }
    } else {
        log::warn!("Snapshots folder does not exist at {}", snapshots.display());
    }

    log::info!(
        "No local ASR snapshot found under {}; downloading from {}",
        root.display(),
        MODEL_BASE_URL
    );
    download::download_default_snapshot(root)
}
