use std::sync::{Mutex, OnceLock};

use serde::Serialize;

pub mod download;
mod paths;
mod verification;

pub use paths::{
    default_model_root, fallback_model_root, invalid_model_files_for_tests,
    model_file_matches_for_tests, resolve_model_dir,
};
pub(crate) use paths::{invalidate_model_verification, resolve_model_dir_with_progress};
pub use verification::{
    receipt_matches_for_tests as verification_receipt_matches_for_tests,
    write_receipt_for_tests as write_verification_receipt_for_tests,
};

#[derive(Clone, Copy)]
pub struct ModelAsset {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub struct ModelSpec {
    pub repository: &'static str,
    pub cache_dir: &'static str,
    pub revision: &'static str,
    pub assets: &'static [ModelAsset],
}

pub const MODEL_SPEC: ModelSpec = ModelSpec {
    repository: "smcleod/nemotron-3.5-asr-streaming-0.6b-int8",
    cache_dir: "models--smcleod--nemotron-3.5-asr-streaming-0.6b-int8",
    revision: "f1f26d22dab5c4eabe6d01b63c906889e7e817d3",
    assets: &[
        ModelAsset {
            name: "config.json",
            size: 2_970,
            sha256: "193c4bba0f21a16c53ccdb6f5586b710b9790f1065fbe718dc9b49b955053308",
        },
        ModelAsset {
            name: "encoder.onnx",
            size: 42_963_073,
            sha256: "a6fd0bbedae97047cb444dba928273b66b9cae36249cf697f4bf7b6f0e167c5d",
        },
        ModelAsset {
            name: "encoder.onnx.data",
            size: 614_649_600,
            sha256: "c2f230b026aa4f29b1b5ce099b2fba853db361773157d478d67127b877f64c42",
        },
        ModelAsset {
            name: "decoder_joint.onnx",
            size: 24_483_962,
            sha256: "7fe1a8c2e247b55bbb8ca917ef64cf60227909c6fe63be2da7ea6fc3858d6a69",
        },
        ModelAsset {
            name: "tokenizer.model",
            size: 406_554,
            sha256: "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291",
        },
    ],
};

pub fn model_base_url() -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}",
        MODEL_SPEC.repository, MODEL_SPEC.revision
    )
}

#[derive(Clone, Debug, Serialize)]
pub struct DownloadProgress {
    pub file_index: usize,
    pub file_count: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub done: bool,
}

static DOWNLOAD_PROGRESS: OnceLock<Mutex<DownloadProgress>> = OnceLock::new();

const MAX_RETRIES: usize = 3;
const RETRY_BACKOFF_SECS: u64 = 2;

fn progress_state() -> &'static Mutex<DownloadProgress> {
    DOWNLOAD_PROGRESS.get_or_init(|| Mutex::new(empty_progress()))
}

fn empty_progress() -> DownloadProgress {
    DownloadProgress {
        file_index: 0,
        file_count: 0,
        downloaded_bytes: 0,
        total_bytes: 0,
        done: false,
    }
}

fn notify_progress(on_progress: &dyn Fn(DownloadProgress)) {
    if let Some(progress) = current_download_progress() {
        on_progress(progress);
    }
}

fn start_tracking(file_count: usize, on_progress: &dyn Fn(DownloadProgress)) {
    if let Ok(mut progress) = progress_state().lock() {
        *progress = DownloadProgress {
            file_count,
            ..empty_progress()
        };
    }
    notify_progress(on_progress);
}

fn set_file_index(file_index: usize, on_progress: &dyn Fn(DownloadProgress)) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = file_index;
        progress.downloaded_bytes = 0;
        progress.total_bytes = 0;
    }
    notify_progress(on_progress);
}

fn update_download_bytes(downloaded: u64, total: u64) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.downloaded_bytes = downloaded;
        progress.total_bytes = total;
    }
}

fn mark_finished(on_progress: &dyn Fn(DownloadProgress)) {
    if let Ok(mut progress) = progress_state().lock() {
        progress.file_index = progress.file_count;
        progress.done = true;
    }
    notify_progress(on_progress);
}

fn current_download_progress() -> Option<DownloadProgress> {
    DOWNLOAD_PROGRESS
        .get()
        .and_then(|mutex| mutex.lock().ok().map(|progress| progress.clone()))
}
