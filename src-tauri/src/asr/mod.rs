pub mod audio_io;
mod decoder;
pub mod download_progress;
mod model_store;
mod recognizer;

use crate::vad::VadModel;
use std::sync::{Arc, OnceLock};
use tauri::AppHandle;

static VAD_MODEL: OnceLock<Arc<VadModel>> = OnceLock::new();

pub fn get_or_init_vad_model(app: &AppHandle) -> Arc<VadModel> {
    VAD_MODEL
        .get_or_init(|| {
            let path = model_store::vad_model_path(app);
            match VadModel::new(&path) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    log::error!("Failed to load VAD model: {e}");
                    panic!("VAD model failed to load at {}: {e}", path.display());
                }
            }
        })
        .clone()
}

pub use model_store::{
    default_model_root, ensure_vad_model, fallback_model_root, resolve_model_dir, vad_model_path,
};
pub use recognizer::{AsrError, AsrModel, Transcript};

pub(crate) use audio_io::{resample_linear, TARGET_SAMPLE_RATE};
pub(crate) use download_progress::{current_download_progress, record_failure, DownloadProgress};
