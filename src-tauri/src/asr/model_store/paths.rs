use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tauri::AppHandle;

use crate::asr::recognizer::AsrError;

use super::{download, verification, DownloadProgress, ModelAsset, MODEL_SPEC};

pub(crate) fn model_file_matches(path: &Path, asset: ModelAsset) -> Result<bool, AsrError> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.len() == asset.size => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(AsrError::io(
                format!("read model asset metadata {}", path.display()),
                error,
            ))
        }
    }

    let mut file = fs::File::open(path)
        .map_err(|error| AsrError::io(format!("open model asset {}", path.display()), error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| AsrError::io(format!("hash model asset {}", path.display()), error))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()) == asset.sha256)
}

fn invalid_model_files(snapshot_dir: &Path) -> Result<Vec<ModelAsset>, AsrError> {
    MODEL_SPEC
        .assets
        .iter()
        .copied()
        .filter_map(
            |asset| match model_file_matches(&snapshot_dir.join(asset.name), asset) {
                Ok(true) => None,
                Ok(false) => Some(Ok(asset)),
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

#[doc(hidden)]
pub fn invalid_model_files_for_tests(snapshot_dir: &Path) -> Result<Vec<String>, AsrError> {
    invalid_model_files(snapshot_dir).map(|assets| {
        assets
            .into_iter()
            .map(|asset| asset.name.to_string())
            .collect()
    })
}

#[doc(hidden)]
pub fn model_file_matches_for_tests(path: &Path, size: u64, sha256: &'static str) -> bool {
    model_file_matches(
        path,
        ModelAsset {
            name: "test",
            size,
            sha256,
        },
    )
    .unwrap_or(false)
}

pub fn default_model_root(app: &AppHandle) -> PathBuf {
    crate::settings::get_custom_model_path(app).unwrap_or_else(fallback_model_root)
}

pub fn fallback_model_root() -> PathBuf {
    let base = dirs_next::cache_dir()
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("huggingface")
        .join("hub")
        .join(MODEL_SPEC.cache_dir)
}

pub fn resolve_model_dir<P: AsRef<Path>>(root: P) -> Result<PathBuf, AsrError> {
    resolve_model_dir_with_progress(root, |_| {})
}

pub(crate) fn invalidate_model_verification(snapshot_dir: &Path) {
    if let Err(error) = verification::remove_receipt(snapshot_dir) {
        log::warn!(
            "Could not invalidate model verification receipt at {}: {error}",
            snapshot_dir.display()
        );
    }
}

pub(crate) fn resolve_model_dir_with_progress<P, F>(
    root: P,
    on_progress: F,
) -> Result<PathBuf, AsrError>
where
    P: AsRef<Path>,
    F: Fn(DownloadProgress),
{
    let root = root.as_ref();
    let snapshot = root.join("snapshots").join(MODEL_SPEC.revision);
    if verification::receipt_matches(&snapshot, MODEL_SPEC.revision, MODEL_SPEC.assets) {
        return Ok(snapshot);
    }

    let invalid = invalid_model_files(&snapshot)?;
    if invalid.is_empty() {
        persist_verification_receipt(&snapshot);
        return Ok(snapshot);
    }

    invalidate_model_verification(&snapshot);
    fs::create_dir_all(&snapshot).map_err(|error| {
        AsrError::io(
            format!("create model snapshot directory {}", snapshot.display()),
            error,
        )
    })?;
    download::download_assets(&snapshot, &invalid, &on_progress)?;
    download::write_revision_ref(root)?;

    let remaining = invalid_model_files(&snapshot)?;
    if remaining.is_empty() {
        persist_verification_receipt(&snapshot);
        return Ok(snapshot);
    }
    Err(AsrError::Integrity(format!(
        "invalid assets after repair: {}",
        remaining
            .iter()
            .map(|asset| asset.name)
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

fn persist_verification_receipt(snapshot_dir: &Path) {
    if let Err(error) =
        verification::write_receipt(snapshot_dir, MODEL_SPEC.revision, MODEL_SPEC.assets)
    {
        log::warn!(
            "Could not cache model verification at {}: {error}",
            snapshot_dir.display()
        );
    }
}
