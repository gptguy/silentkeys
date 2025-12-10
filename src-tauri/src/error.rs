use thiserror::Error;

use crate::asr::AsrError;
use crate::recording::RecordingError;
use crate::vad::VadError;

/// Unified app errors.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Recording: {0}")]
    Recording(#[from] RecordingError),

    #[error("ASR: {0}")]
    Asr(#[from] AsrError),

    #[error("VAD: {0}")]
    Vad(#[from] VadError),

    #[error("Audio: {0}")]
    Audio(#[from] AudioError),

    #[error("Settings: {0}")]
    Settings(String),
}

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// Audio processing errors
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(usize),

    #[error("Invalid frame duration: must be positive")]
    InvalidFrameDuration,

    #[error("Failed to create resampler: {0}")]
    ResamplerCreation(String),

    #[error("Resampler processing failed: {0}")]
    ResamplerProcessing(String),

    #[error("VAD error: {0}")]
    VadError(String),
}
