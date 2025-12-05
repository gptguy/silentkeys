mod audio_io;
mod decoder;
mod download_progress;
mod model_store;
mod recognizer;

#[cfg(test)]
mod tests;

// Public API
pub use model_store::{default_model_root, fallback_model_root, resolve_model_dir};
pub use recognizer::{AsrError, AsrModel, Transcript};

// Internal API for other crate modules
pub(crate) use audio_io::{resample_linear, TARGET_SAMPLE_RATE};
pub(crate) use download_progress::{current_download_progress, record_failure, DownloadProgress};
