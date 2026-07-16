use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use super::ModelAsset;

const RECEIPT_FILE: &str = ".silentkeys-verified.json";
const RECEIPT_SCHEMA_VERSION: u32 = 1;

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

pub(super) fn receipt_matches(snapshot_dir: &Path, revision: &str, assets: &[ModelAsset]) -> bool {
    let stored = fs::read(receipt_path(snapshot_dir))
        .ok()
        .and_then(|data| serde_json::from_slice::<VerificationReceipt>(&data).ok());
    let current = capture_receipt(snapshot_dir, revision, assets).ok();
    stored.is_some() && stored == current
}

pub(super) fn write_receipt(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[ModelAsset],
) -> io::Result<()> {
    let receipt = capture_receipt(snapshot_dir, revision, assets)?;
    let data = serde_json::to_vec(&receipt).map_err(io::Error::other)?;
    let destination = receipt_path(snapshot_dir);
    let temporary = temporary_receipt_path(snapshot_dir);

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&data)?;
    file.sync_all()?;

    if let Err(error) = fs::rename(&temporary, &destination) {
        if error.kind() != io::ErrorKind::AlreadyExists {
            return Err(error);
        }
        fs::remove_file(&destination)?;
        fs::rename(&temporary, &destination)?;
    }
    Ok(())
}

pub(super) fn remove_receipt(snapshot_dir: &Path) -> io::Result<()> {
    match fs::remove_file(receipt_path(snapshot_dir)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn capture_receipt(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[ModelAsset],
) -> io::Result<VerificationReceipt> {
    let assets = assets
        .iter()
        .map(|asset| capture_asset(snapshot_dir, *asset))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(VerificationReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        revision: revision.to_string(),
        assets,
    })
}

fn capture_asset(snapshot_dir: &Path, asset: ModelAsset) -> io::Result<VerifiedAsset> {
    let metadata = fs::metadata(snapshot_dir.join(asset.name))?;
    if !metadata.is_file() || metadata.len() != asset.size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model asset metadata does not match: {}", asset.name),
        ));
    }
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
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
) -> bool {
    let assets = test_assets(assets);
    receipt_matches(snapshot_dir, revision, &assets)
}

#[doc(hidden)]
pub fn write_receipt_for_tests(
    snapshot_dir: &Path,
    revision: &str,
    assets: &[(&'static str, u64, &'static str)],
) -> io::Result<()> {
    let assets = test_assets(assets);
    write_receipt(snapshot_dir, revision, &assets)
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
