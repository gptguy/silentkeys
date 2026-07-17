pub mod pipeline;

pub use pipeline::StreamingPipeline;

use serde::Serialize;

use crate::asr::AsrError;
use crate::errors::UserFacing;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
pub enum TranscriptionUpdate {
    Append(String),
    Replace(String),
}

/// Delivers committed transcription updates to a UI or typing sink.
pub trait UpdateSink: Fn(TranscriptionUpdate) -> Result<(), String> + Send + 'static {}

impl<F: Fn(TranscriptionUpdate) -> Result<(), String> + Send + 'static> UpdateSink for F {}

#[derive(thiserror::Error, Debug)]
pub enum StreamingError {
    #[error("streaming transcription is already running")]
    AlreadyRunning,
    #[error("{0} lock failed")]
    LockFailed(&'static str),
    #[error("speech model is not ready")]
    ModelNotReady,
    #[error("speech model is not loaded")]
    ModelNotLoaded,
    #[error("could not start streaming worker: {0}")]
    WorkerStart(#[source] std::io::Error),
    #[error("streaming speech recognition failed: {0}")]
    Decode(#[from] AsrError),
    #[error("transcription output failed: {0}")]
    Output(String),
    #[error("streaming worker thread panicked")]
    WorkerPanicked,
}

impl UserFacing for StreamingError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::AlreadyRunning => "Streaming transcription is already running.",
            Self::ModelNotReady | Self::ModelNotLoaded => {
                "The speech model is not ready. Please wait and try again."
            }
            Self::LockFailed(_)
            | Self::WorkerStart(_)
            | Self::Decode(_)
            | Self::Output(_)
            | Self::WorkerPanicked => "Streaming transcription failed. Please try recording again.",
        }
    }
}
