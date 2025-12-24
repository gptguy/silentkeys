use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::Serialize;

use crate::asr::recognizer::AsrError;

use super::{MAX_RETRIES, MODEL_BASE_URL, MODEL_FILES, RETRY_BACKOFF_SECS};

pub(crate) fn download_missing_files(
    snapshot_dir: &Path,
    missing_files: &[String],
) -> Result<(), AsrError> {
    if missing_files.is_empty() {
        return Ok(());
    }

    start_tracking(missing_files.len());

    let result: Result<(), AsrError> = (|| {
        for (index, file) in missing_files.iter().enumerate() {
            set_file_index(index + 1);
            let dest = snapshot_dir.join(file);

            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }

            let url = format!("{MODEL_BASE_URL}/{file}");
            download_asset(&url, &dest)?;
        }
        Ok(())
    })();

    match result {
        Ok(()) => {
            mark_finished();
            log::info!("Repaired ASR snapshot at {}", snapshot_dir.display());
            Ok(())
        }
        Err(err) => {
            log::error!("ASR model repair download failed: {}", err);
            record_failure(err.user_message().to_string());
            Err(err)
        }
    }
}

pub(crate) fn download_default_snapshot(root: &Path) -> Result<PathBuf, AsrError> {
    start_tracking(MODEL_FILES.len());

    let result: Result<PathBuf, AsrError> = (|| {
        let snapshots = root.join("snapshots");
        fs::create_dir_all(&snapshots)?;

        let download_dir = snapshots.join("downloaded");
        fs::create_dir_all(&download_dir)?;

        for (index, file) in MODEL_FILES.iter().enumerate() {
            let file_index = index + 1;
            let dest = download_dir.join(file);

            set_file_index(file_index);

            if dest.exists() {
                continue;
            }

            let url = format!("{MODEL_BASE_URL}/{file}");
            download_asset(&url, &dest)?;
        }

        write_refs_main(root, &download_dir)?;

        Ok(download_dir)
    })();

    match result {
        Ok(path) => {
            mark_finished();
            Ok(path)
        }
        Err(err) => {
            log::error!("ASR model download failed: {err}");
            record_failure(err.user_message().to_string());
            Err(err)
        }
    }
}

fn download_asset(url: &str, dest: &Path) -> Result<(), AsrError> {
    let tmp = dest.with_extension("download");
    let mut last_err: Option<AsrError> = None;

    let config = ureq::config::Config::builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build();
    let agent = ureq::Agent::new_with_config(config);

    for attempt in 1..=MAX_RETRIES {
        log::info!(
            "Downloading model asset to {} from {url} (attempt {attempt}/{MAX_RETRIES})",
            dest.display()
        );

        match try_download_resumable(&agent, url, &tmp, dest) {
            Ok(()) => return Ok(()),
            Err(err) => {
                log::warn!("Download attempt {} failed: {}", attempt, err);
                last_err = Some(err);

                if attempt < MAX_RETRIES {
                    std::thread::sleep(Duration::from_secs(RETRY_BACKOFF_SECS * attempt as u64));
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| AsrError::Download(format!("{url}: failed to download"))))
}

fn write_refs_main(root: &Path, snapshot_dir: &Path) -> Result<(), AsrError> {
    let refs_dir = root.join("refs");
    fs::create_dir_all(&refs_dir)?;

    let snapshot_name = snapshot_dir
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| AsrError::Download("invalid snapshot directory name".to_string()))?;

    let refs_main = refs_dir.join("main");
    fs::write(&refs_main, snapshot_name)
        .map_err(|e| AsrError::Download(format!("write refs/main: {e}")))?;

    Ok(())
}

fn try_download_resumable(
    agent: &ureq::Agent,
    url: &str,
    tmp: &Path,
    dest: &Path,
) -> Result<(), AsrError> {
    let current_len = if tmp.exists() {
        fs::metadata(tmp).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut request = agent.get(url);
    if current_len > 0 {
        request = request.header("Range", &format!("bytes={}-", current_len));
    }

    let response = request
        .call()
        .map_err(|e| AsrError::Download(format!("{url}: request failed: {e}")))?;

    let status = response.status();
    if !(200..300).contains(&status.as_u16()) {
        return Err(AsrError::Download(format!(
            "{url}: unexpected status {status}"
        )));
    }
    let total_size = if status == 206 {
        let content_len = response
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        current_len + content_len
    } else {
        response
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let mut file = if status == 206 {
        log::debug!("Resuming download from byte {}", current_len);
        fs::OpenOptions::new().create(true).append(true).open(tmp)?
    } else {
        if current_len > 0 {
            log::warn!(
                "Server does not support resuming or file changed (status {}), restarting download.",
                status
            );
        }
        fs::File::create(tmp)?
    };

    update_download_bytes(if status == 206 { current_len } else { 0 }, total_size);

    let mut downloaded = if status == 206 { current_len } else { 0 };
    let mut buffer = [0; 8192];

    let mut reader = response.into_body().into_reader();
    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| AsrError::Download(format!("{url}: read failed: {e}")))?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .map_err(|e| AsrError::Download(format!("{url}: write failed: {e}")))?;

        downloaded += bytes_read as u64;
        update_download_bytes(downloaded, total_size);
    }

    if total_size > 0 && downloaded != total_size {
        return Err(AsrError::Download(format!(
            "Incomplete download: expected {} bytes, got {}",
            total_size, downloaded
        )));
    }

    fs::rename(tmp, dest)?;
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct DownloadProgress {
    pub file_index: usize,
    pub file_count: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub done: bool,
    pub error: Option<String>,
}

static DOWNLOAD_PROGRESS: OnceLock<Mutex<DownloadProgress>> = OnceLock::new();

fn progress_state() -> &'static Mutex<DownloadProgress> {
    DOWNLOAD_PROGRESS.get_or_init(|| {
        Mutex::new(DownloadProgress {
            file_index: 0,
            file_count: 0,
            downloaded_bytes: 0,
            total_bytes: 0,
            done: false,
            error: None,
        })
    })
}

pub fn start_tracking(file_count: usize) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = 0;
        progress.file_count = file_count;
        progress.downloaded_bytes = 0;
        progress.total_bytes = 0;
        progress.done = false;
        progress.error = None;
    }
}

pub fn set_file_index(file_index: usize) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = file_index;
        progress.downloaded_bytes = 0;
        progress.total_bytes = 0;
    }
}

pub(crate) fn update_download_bytes(downloaded: u64, total: u64) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.downloaded_bytes = downloaded;
        progress.total_bytes = total;
    }
}

pub fn mark_finished() {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = progress.file_count;
        progress.done = true;
    }
}

pub fn record_failure(error: String) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.error = Some(error);
        progress.done = true;
    }
}

pub fn current_download_progress() -> Option<DownloadProgress> {
    DOWNLOAD_PROGRESS
        .get()
        .and_then(|mutex| mutex.lock().ok().map(|progress| progress.clone()))
}
