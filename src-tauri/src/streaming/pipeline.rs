use std::sync::{mpsc::Receiver, Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crate::asr::{AsrModel, STREAM_CHUNK_SAMPLES};
use crate::audio_processing::AudioFrame;
use crate::streaming::{StreamingError, TranscriptionUpdate, UpdateSink};

type Worker = JoinHandle<Result<(), StreamingError>>;

/// Owns at most one decoding worker; the worker slot is the running/idle state.
#[derive(Default)]
pub struct StreamingPipeline {
    worker: Mutex<Option<Worker>>,
}

impl StreamingPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves the audio receiver and output sink to one decoder worker. Dropping
    /// every sender closes the stream; `finish` then joins the worker after its flush.
    pub fn start(
        &self,
        audio: Receiver<AudioFrame>,
        model: Arc<RwLock<Option<AsrModel>>>,
        on_update: impl UpdateSink,
    ) -> Result<(), StreamingError> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| StreamingError::LockFailed("streaming worker"))?;
        if worker.is_some() {
            return Err(StreamingError::AlreadyRunning);
        }

        *worker = Some(
            thread::Builder::new()
                .name("streaming-decode".to_string())
                .spawn(move || Self::run(audio, model, on_update))
                .map_err(StreamingError::WorkerStart)?,
        );
        Ok(())
    }

    fn run(
        audio: Receiver<AudioFrame>,
        model: Arc<RwLock<Option<AsrModel>>>,
        on_update: impl UpdateSink,
    ) -> Result<(), StreamingError> {
        let mut pending = Vec::with_capacity(STREAM_CHUNK_SAMPLES + 512);
        while let Ok(frame) = audio.recv() {
            pending.extend_from_slice(&frame.samples);
            while pending.len() >= STREAM_CHUNK_SAMPLES {
                let text = Self::with_model(&model, |model| {
                    model.advance_streaming(&pending[..STREAM_CHUNK_SAMPLES])
                })?;
                Self::emit(text, &on_update)?;
                let remaining = pending.len() - STREAM_CHUNK_SAMPLES;
                pending.copy_within(STREAM_CHUNK_SAMPLES.., 0);
                pending.truncate(remaining);
            }
        }
        if !pending.is_empty() {
            let text = Self::with_model(&model, |model| model.advance_streaming(&pending))?;
            Self::emit(text, &on_update)?;
        }

        let text = Self::with_model(&model, AsrModel::finish_streaming)?;
        Self::emit(text, &on_update)
    }

    fn with_model<T>(
        model: &RwLock<Option<AsrModel>>,
        operation: impl FnOnce(&mut AsrModel) -> Result<T, crate::asr::AsrError>,
    ) -> Result<T, StreamingError> {
        let mut guard = model
            .write()
            .map_err(|_| StreamingError::LockFailed("speech model"))?;
        operation(guard.as_mut().ok_or(StreamingError::ModelNotLoaded)?).map_err(Into::into)
    }

    fn emit(text: String, on_update: &impl UpdateSink) -> Result<(), StreamingError> {
        if text.is_empty() {
            return Ok(());
        }
        on_update(TranscriptionUpdate::Append(text)).map_err(StreamingError::Output)?;
        Ok(())
    }

    pub fn finish(&self) -> Result<(), StreamingError> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| StreamingError::LockFailed("streaming worker"))?
            .take();
        match worker {
            Some(worker) => worker.join().map_err(|_| StreamingError::WorkerPanicked)?,
            None => Ok(()),
        }
    }
}
