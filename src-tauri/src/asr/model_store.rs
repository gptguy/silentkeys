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

pub fn resolve_model_dir<P: AsRef<Path>>(root: P) -> Result<PathBuf, AsrError> {
    let root = root.as_ref();
    log::info!("resolve_model_dir: checking root {}", root.display());

    let refs_main = root.join("refs").join("main");

    if refs_main.exists() {
        let commit = fs::read_to_string(&refs_main)?.trim().to_string();
        let snap = root.join("snapshots").join(&commit);
        log::info!(
            "Found refs/main: {}, checking snapshot at {}",
            commit,
            snap.display()
        );
        if snap.is_dir() {
            log::info!("Snapshot directory exists and is valid.");
            return Ok(snap);
        } else {
            log::warn!("Snapshot directory from refs/main does NOT exist or is not a dir.");
        }
    } else {
        log::info!("refs/main not found at {}", refs_main.display());
    }

    let snapshots = root.join("snapshots");
    if snapshots.is_dir() {
        log::info!("Scanning snapshots dir: {}", snapshots.display());
        let mut newest: Option<(SystemTime, PathBuf)> = None;
        for entry in fs::read_dir(&snapshots)? {
            let entry = entry?;
            let path = entry.path();
            log::debug!("Found entry: {}", path.display());

            if !entry.file_type()?.is_dir() {
                log::debug!("Skipping non-dir: {}", path.display());
                continue;
            }

            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);

            match &mut newest {
                Some((ts, best)) if modified > *ts => {
                    log::debug!(
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
            return Ok(path);
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

        let refs_dir = root.join("refs");
        fs::create_dir_all(&refs_dir)?;
        let refs_main = refs_dir.join("main");
        if !refs_main.exists() {
            if let Some(name) = download_dir.file_name().and_then(|n| n.to_str()) {
                fs::write(&refs_main, name)
                    .map_err(|e| AsrError::Download(format!("write refs/main: {e}")))?;
            }
        }

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

    for attempt in 1..=MAX_RETRIES {
        if tmp.exists() {
            let _ = fs::remove_file(&tmp);
        }

        log::info!("Downloading model asset from {url} (attempt {attempt}/{MAX_RETRIES})...");

        match try_download_once(url, &tmp, dest) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_err = Some(err);

                if attempt < MAX_RETRIES {
                    std::thread::sleep(Duration::from_secs(RETRY_BACKOFF_SECS * attempt as u64));
                } else if tmp.exists() {
                    let _ = fs::remove_file(&tmp);
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| AsrError::Download(format!("{url}: failed to download"))))
}

fn try_download_once(url: &str, tmp: &Path, dest: &Path) -> Result<(), AsrError> {
    let response =
        reqwest::blocking::get(url).map_err(|e| AsrError::Download(format!("{url}: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(AsrError::Download(format!(
            "{url}: unexpected status {status}"
        )));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let mut file = fs::File::create(tmp)?;
    let mut reader = response;
    let mut buffer = [0; 8192];

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

    fs::rename(tmp, dest)?;

    Ok(())
}
