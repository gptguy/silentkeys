use serde::Serialize;
use std::sync::{Mutex, OnceLock};

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

pub(crate) fn start_tracking(file_count: usize) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = 0;
        progress.file_count = file_count;
        progress.downloaded_bytes = 0;
        progress.total_bytes = 0;
        progress.done = false;
        progress.error = None;
    }
}

pub(crate) fn set_file_index(file_index: usize) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = file_index;
        // Reset bytes for the new file
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

pub(crate) fn mark_finished() {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = progress.file_count;
        progress.done = true;
    }
}

pub(crate) fn record_failure(error: String) {
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
