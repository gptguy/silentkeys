use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTimeError, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::ModelAsset;

const RECEIPT_FILE: &str = ".silentkeys-verified.json";
const RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(thiserror::Error, Debug)]
pub(super) enum VerificationError {
    #[error("{operation} {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("serialize model verification receipt {}: {source}", path.display())]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "model asset metadata does not match {} (expected a {expected_size}-byte file)",
        path.display()
    )]
    AssetMetadata { path: PathBuf, expected_size: u64 },
    #[error("read model asset modification time {}: {source}", path.display())]
    ModifiedTime {
        path: PathBuf,
        #[source]
        source: SystemTimeError,
    },
}

impl VerificationError {
    fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }

    fn is_cache_miss(&self) -> bool {
        match self {
            Self::AssetMetadata { .. } => true,
            Self::Io { source, .. } => source.kind() == io::ErrorKind::NotFound,
            Self::Serialize { .. } | Self::ModifiedTime { .. } => false,
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
struct VerificationReceipt {
    schema_version: u32,
    revision: String,
    assets: Vec<VerifiedAsset>,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
struct VerifiedAsset {
    name: String,
    size: u64,
    sha256: String,
    modified_seconds: u64,
    modified_nanoseconds: u32,
}

pub(super) fn receipt_matches(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[ModelAsset],
) -> Result<bool, VerificationError> {
    let path = receipt_path(snapshot_dir);
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(VerificationError::io(
                "read verification receipt",
                &path,
                error,
            ))
        }
    };
    let stored = match serde_json::from_slice::<VerificationReceipt>(&data) {
        Ok(receipt) => receipt,
        Err(_) => return Ok(false),
    };
    match capture_receipt(snapshot_dir, revision, assets) {
        Ok(current) => Ok(stored == current),
        Err(error) if error.is_cache_miss() => Ok(false),
        Err(error) => Err(error),
    }
}

pub(super) fn write_receipt(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[ModelAsset],
) -> Result<(), VerificationError> {
    let receipt = capture_receipt(snapshot_dir, revision, assets)?;
    let destination = receipt_path(snapshot_dir);
    let temporary = temporary_receipt_path(snapshot_dir);
    let data = serde_json::to_vec(&receipt).map_err(|source| VerificationError::Serialize {
        path: destination.clone(),
        source,
    })?;

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| {
            VerificationError::io("open temporary verification receipt", &temporary, error)
        })?;
    file.write_all(&data).map_err(|error| {
        VerificationError::io("write temporary verification receipt", &temporary, error)
    })?;
    file.sync_all().map_err(|error| {
        VerificationError::io("sync temporary verification receipt", &temporary, error)
    })?;

    if let Err(error) = fs::rename(&temporary, &destination) {
        if error.kind() != io::ErrorKind::AlreadyExists {
            return Err(VerificationError::io(
                "install verification receipt",
                &destination,
                error,
            ));
        }
        fs::remove_file(&destination).map_err(|error| {
            VerificationError::io("replace verification receipt", &destination, error)
        })?;
        fs::rename(&temporary, &destination).map_err(|error| {
            VerificationError::io("install verification receipt", &destination, error)
        })?;
    }
    Ok(())
}

pub(super) fn remove_receipt(snapshot_dir: &Path) -> Result<(), VerificationError> {
    let path = receipt_path(snapshot_dir);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(VerificationError::io(
            "remove verification receipt",
            &path,
            error,
        )),
    }
}

fn capture_receipt(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[ModelAsset],
) -> Result<VerificationReceipt, VerificationError> {
    let assets = assets
        .iter()
        .map(|asset| capture_asset(snapshot_dir, *asset))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VerificationReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        revision: revision.to_string(),
        assets,
    })
}

fn capture_asset(
    snapshot_dir: &Path,
    asset: ModelAsset,
) -> Result<VerifiedAsset, VerificationError> {
    let path = snapshot_dir.join(asset.name);
    let metadata = fs::metadata(&path)
        .map_err(|error| VerificationError::io("read model asset metadata", &path, error))?;
    if !metadata.is_file() || metadata.len() != asset.size {
        return Err(VerificationError::AssetMetadata {
            path,
            expected_size: asset.size,
        });
    }
    let modified = metadata
        .modified()
        .map_err(|error| VerificationError::io("read model asset modification time", &path, error))?
        .duration_since(UNIX_EPOCH)
        .map_err(|source| VerificationError::ModifiedTime {
            path: path.clone(),
            source,
        })?;
    Ok(VerifiedAsset {
        name: asset.name.to_string(),
        size: asset.size,
        sha256: asset.sha256.to_string(),
        modified_seconds: modified.as_secs(),
        modified_nanoseconds: modified.subsec_nanos(),
    })
}

fn receipt_path(snapshot_dir: &Path) -> PathBuf {
    snapshot_dir.join(RECEIPT_FILE)
}

fn temporary_receipt_path(snapshot_dir: &Path) -> PathBuf {
    snapshot_dir.join(format!("{RECEIPT_FILE}.tmp"))
}

#[doc(hidden)]
pub fn receipt_matches_for_tests(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[(&'static str, u64, &'static str)],
) -> Result<bool, String> {
    let assets = test_assets(assets);
    receipt_matches(snapshot_dir, revision, &assets).map_err(|error| error.to_string())
}

#[doc(hidden)]
pub fn write_receipt_for_tests(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[(&'static str, u64, &'static str)],
) -> Result<(), String> {
    let assets = test_assets(assets);
    write_receipt(snapshot_dir, revision, &assets).map_err(|error| error.to_string())
}

fn test_assets(assets: &[(&'static str, u64, &'static str)]) -> Vec<ModelAsset> {
    assets
        .iter()
        .map(|(name, size, sha256)| ModelAsset {
            name,
            size: *size,
            sha256,
        })
        .collect()
}
