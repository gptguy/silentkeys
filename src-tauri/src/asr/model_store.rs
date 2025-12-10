use crate::asr::download_progress::{
    mark_finished, record_failure, set_file_index, start_tracking, update_download_bytes,
};
use crate::asr::recognizer::AsrError;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tauri::AppHandle;

const MODEL_BASE_URL: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";

const VAD_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx";

const VAD_MODEL_FILE: &str = "silero_vad_model.onnx";

const MODEL_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "encoder-model.onnx",
    "decoder_joint-model.onnx",
    "nemo128.onnx",
    "vocab.txt",
];

const MAX_RETRIES: usize = 3;
const RETRY_BACKOFF_SECS: u64 = 2;

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

fn download_missing_files(snapshot_dir: &Path, missing_files: &[String]) -> Result<(), AsrError> {
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

pub fn vad_model_path(app: &AppHandle) -> PathBuf {
    let root = default_model_root(app);
    root.join("snapshots")
        .join("downloaded")
        .join(VAD_MODEL_FILE)
}

pub fn ensure_vad_model(app: &AppHandle) -> Result<PathBuf, AsrError> {
    let path = vad_model_path(app);
    if path.exists() {
        return Ok(path);
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    start_tracking(1);
    set_file_index(1);

    log::info!("Downloading VAD model...");
    let result = download_asset(VAD_MODEL_URL, &path);

    match result {
        Ok(()) => {
            log::info!("VAD model downloaded to {}", path.display());
            mark_finished();
            Ok(path)
        }
        Err(e) => {
            record_failure(e.user_message().to_string());
            Err(e)
        }
    }
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
        return download_default_snapshot(root);
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

    match download_missing_files(&snapshot_dir, &missing) {
        Ok(()) => Ok(snapshot_dir),
        Err(err) => {
            log::warn!(
                "Snapshot repair failed, falling back to fresh download: {}",
                err
            );
            download_default_snapshot(root)
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
    download_default_snapshot(root)
}

fn download_default_snapshot(root: &Path) -> Result<PathBuf, AsrError> {
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

        let vad_dest = download_dir.join(VAD_MODEL_FILE);
        if !vad_dest.exists() {
            log::info!("Downloading VAD model...");
            download_asset(VAD_MODEL_URL, &vad_dest)?;
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

    let agent = ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(30))
        .timeout_write(Duration::from_secs(30))
        .build();

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
    // 1. Check current size of tmp file
    let current_len = if tmp.exists() {
        fs::metadata(tmp).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    // 2. Make Request
    let mut request = agent.get(url);
    if current_len > 0 {
        request = request.set("Range", &format!("bytes={}-", current_len));
    }

    let response = request
        .call()
        .map_err(|e| AsrError::Download(format!("{url}: request failed: {e}")))?;

    let status = response.status();
    // ureq returns success for 2xx.
    if !(200..300).contains(&status) {
        return Err(AsrError::Download(format!(
            "{url}: unexpected status {status}"
        )));
    }

    // 3. Handle Response
    let total_size = if status == 206 {
        // Partial Content
        let content_len = response
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        current_len + content_len
    } else {
        // If server didn't respect Range (returned 200), we overwrite
        response
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let mut file = if status == 206 {
        log::debug!("Resuming download from byte {}", current_len);
        fs::OpenOptions::new().create(true).append(true).open(tmp)?
    } else {
        if current_len > 0 {
            log::warn!("Server does not support resuming or file changed (status {}), restarting download.", status);
        }
        fs::File::create(tmp)?
    };

    // Update progress bar
    update_download_bytes(if status == 206 { current_len } else { 0 }, total_size);

    let mut downloaded = if status == 206 { current_len } else { 0 };
    let mut buffer = [0; 8192];

    let mut reader = response.into_reader();
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

    // Verify size if known
    if total_size > 0 && downloaded != total_size {
        return Err(AsrError::Download(format!(
            "Incomplete download: expected {} bytes, got {}",
            total_size, downloaded
        )));
    }

    fs::rename(tmp, dest)?;
    Ok(())
}
