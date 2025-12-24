pub mod decoder;

mod model_store;
mod recognizer;

pub use model_store::{
    default_model_root, fallback_model_root, missing_model_files_for_tests, resolve_model_dir,
};
pub use recognizer::{AsrError, AsrModel, InferenceConfig, Transcript};

pub use model_store::download::{
    current_download_progress, mark_finished, record_failure, set_file_index, start_tracking,
    DownloadProgress,
};

pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;
pub(crate) const WINDOW_SIZE_SEC: f32 = 0.01;
pub(crate) const SUBSAMPLING_FACTOR: usize = 8;
pub(crate) const FRAME_DURATION_SEC: f32 = WINDOW_SIZE_SEC * SUBSAMPLING_FACTOR as f32;
pub(crate) const FRAME_DURATION_MS: i64 = (FRAME_DURATION_SEC * 1000.0) as i64;
