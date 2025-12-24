pub mod audio_io;
pub mod decoder;
pub mod download_progress;
mod model_store;
mod recognizer;

use tauri::AppHandle;

pub fn get_or_init_vad_model(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    model_store::ensure_vad_model(app).map_err(|e| e.user_message().to_string())
}

pub use model_store::{
    default_model_root, ensure_vad_model, fallback_model_root, missing_model_files_for_tests,
    resolve_model_dir,
};
pub use recognizer::{AsrError, AsrModel, InferenceConfig, Transcript};

pub(crate) use audio_io::TARGET_SAMPLE_RATE;
pub(crate) use download_progress::{current_download_progress, record_failure, DownloadProgress};

pub(crate) const WINDOW_SIZE_SEC: f32 = 0.01;
pub(crate) const SUBSAMPLING_FACTOR: usize = 8;
pub(crate) const FRAME_DURATION_SEC: f32 = WINDOW_SIZE_SEC * SUBSAMPLING_FACTOR as f32;
