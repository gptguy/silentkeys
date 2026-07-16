use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use crate::asr::recognizer::AsrError;

use super::paths::model_file_matches;
use super::{
    mark_finished, model_base_url, notify_progress, set_file_index, start_tracking,
    update_download_bytes, DownloadProgress, ModelAsset, MAX_RETRIES, MODEL_SPEC,
    RETRY_BACKOFF_SECS,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSFER_TIMEOUT: Duration = Duration::from_secs(4 * 60 * 60);
const DOWNLOAD_BUFFER_BYTES: usize = 64 * 1024;
const PROGRESS_REPORT_BYTES: u64 = 1024 * 1024;

pub(crate) fn download_assets(
    snapshot_dir: &Path,
    assets: &[ModelAsset],
    on_progress: &dyn Fn(DownloadProgress),
) -> Result<(), AsrError> {
    if assets.is_empty() {
        return Ok(());
    }
    start_tracking(assets.len(), on_progress);

    assets.iter().enumerate().try_for_each(|(index, asset)| {
        set_file_index(index + 1, on_progress);
        download_asset(
            &model_base_url(),
            *asset,
            &snapshot_dir.join(asset.name),
            on_progress,
        )
    })?;
    mark_finished(on_progress);
    Ok(())
}

pub(crate) fn write_revision_ref(root: &Path) -> Result<(), AsrError> {
    let refs_dir = root.join("refs");
    fs::create_dir_all(&refs_dir).map_err(|error| {
        AsrError::io(
            format!("create model refs directory {}", refs_dir.display()),
            error,
        )
    })?;
    let revision_ref = refs_dir.join("main");
    fs::write(&revision_ref, MODEL_SPEC.revision).map_err(|error| {
        AsrError::io(
            format!("write model revision ref {}", revision_ref.display()),
            error,
        )
    })
}

fn download_asset(
    base_url: &str,
    asset: ModelAsset,
    dest: &Path,
    on_progress: &dyn Fn(DownloadProgress),
) -> Result<(), AsrError> {
    if model_file_matches(dest, asset)? {
        return Ok(());
    }
    if dest.exists() {
        fs::remove_file(dest).map_err(|error| {
            AsrError::io(
                format!("remove invalid model asset {}", dest.display()),
                error,
            )
        })?;
    }

    let tmp = dest.with_extension("download");
    if fs::metadata(&tmp)
        .map(|metadata| metadata.len() >= asset.size)
        .unwrap_or(false)
    {
        fs::remove_file(&tmp).map_err(|error| {
            AsrError::io(
                format!("remove invalid partial download {}", tmp.display()),
                error,
            )
        })?;
    }

    let config = ureq::config::Config::builder()
        .timeout_global(Some(TRANSFER_TIMEOUT))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let url = format!("{base_url}/{}", asset.name);
    let mut last_error = None;

    for attempt in 1..=MAX_RETRIES {
        log::info!(
            "Downloading model asset {} (attempt {attempt}/{MAX_RETRIES})",
            asset.name
        );
        match try_download_resumable(&agent, &url, &tmp, asset.size, on_progress) {
            Ok(()) if model_file_matches(&tmp, asset)? => {
                fs::rename(&tmp, dest).map_err(|error| {
                    AsrError::io(
                        format!(
                            "install model asset {} as {}",
                            tmp.display(),
                            dest.display()
                        ),
                        error,
                    )
                })?;
                return Ok(());
            }
            Ok(()) => {
                let _ = fs::remove_file(&tmp);
                last_error = Some(AsrError::Integrity(asset.name.to_string()));
            }
            Err(error) => last_error = Some(error),
        }
        if attempt < MAX_RETRIES {
            std::thread::sleep(Duration::from_secs(RETRY_BACKOFF_SECS * attempt as u64));
        }
    }

    Err(last_error.unwrap_or_else(|| AsrError::Download(url)))
}

fn try_download_resumable(
    agent: &ureq::Agent,
    url: &str,
    tmp: &Path,
    expected_size: u64,
    on_progress: &dyn Fn(DownloadProgress),
) -> Result<(), AsrError> {
    let current_len = fs::metadata(tmp)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut request = agent.get(url);
    if current_len > 0 {
        request = request.header("Range", &format!("bytes={current_len}-"));
    }

    let response = request
        .call()
        .map_err(|error| AsrError::Download(format!("{url}: request failed: {error}")))?;
    let resumed = response.status().as_u16() == 206;
    let mut file = if resumed {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(tmp)
            .map_err(|error| {
                AsrError::io(format!("open partial download {}", tmp.display()), error)
            })?
    } else {
        fs::File::create(tmp).map_err(|error| {
            AsrError::io(format!("create partial download {}", tmp.display()), error)
        })?
    };
    let mut downloaded = if resumed { current_len } else { 0 };
    let mut last_reported = downloaded;
    update_download_bytes(downloaded, expected_size);
    notify_progress(on_progress);

    let mut reader = response.into_body().into_reader();
    let mut buffer = [0_u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| AsrError::Download(format!("{url}: read failed: {error}")))?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count]).map_err(|error| {
            AsrError::io(format!("write partial download {}", tmp.display()), error)
        })?;
        downloaded += count as u64;
        update_download_bytes(downloaded, expected_size);
        if downloaded == expected_size
            || downloaded.saturating_sub(last_reported) >= PROGRESS_REPORT_BYTES
        {
            notify_progress(on_progress);
            last_reported = downloaded;
        }
    }

    if downloaded != expected_size {
        return Err(AsrError::Download(format!(
            "{}: expected {} bytes, received {}",
            url, expected_size, downloaded
        )));
    }
    file.sync_all().map_err(|error| {
        AsrError::io(
            format!("sync completed model download {}", tmp.display()),
            error,
        )
    })?;
    Ok(())
}
