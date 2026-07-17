use crate::engine::{EngineError, SpeechEngine};
use crate::errors::UserFacing;
use crate::recording::{RecordingError, RecordingReservation};
use crate::streaming::{StreamingError, UpdateSink};

#[derive(thiserror::Error, Debug)]
pub enum DictationError {
    #[error(transparent)]
    Recording(#[from] RecordingError),
    #[error(transparent)]
    Streaming(#[from] StreamingError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("dictation output failed: {0}")]
    Output(String),
}

impl UserFacing for DictationError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::Recording(error) => error.user_message(),
            Self::Streaming(error) => error.user_message(),
            Self::Engine(error) => error.user_message(),
            Self::Output(_) => "Could not deliver the transcription. Please try again.",
        }
    }
}

impl SpeechEngine {
    pub fn reserve_dictation(&self) -> Result<RecordingReservation, DictationError> {
        Ok(self.recorder().reserve()?)
    }

    pub fn start_dictation(
        &self,
        reservation: RecordingReservation,
        on_update: impl UpdateSink,
    ) -> Result<(), DictationError> {
        let streaming = crate::settings::get_settings(self.app()).streaming_enabled;
        self.reset_model_state();
        let streaming_tx = if streaming {
            Some(self.start_streaming(on_update)?)
        } else {
            None
        };

        if let Err(error) = self.recorder().start(reservation, streaming_tx) {
            let _ = self.finish_streaming();
            return Err(error.into());
        }
        log::info!("Dictation started (streaming={streaming})");
        Ok(())
    }

    /// Streaming minimizes perceived latency; the final offline pass is the
    /// canonical transcript and corrects any divergent partial output.
    pub fn finish_dictation<F>(&self, on_text: F) -> Result<(), DictationError>
    where
        F: FnOnce(String) -> Result<(), String>,
    {
        let audio_result = self.recorder().stop();
        if let Err(error) = self.finish_streaming() {
            log::warn!("Streaming failed; using final offline transcription: {error}");
        }
        let audio = match audio_result {
            Ok(audio) => audio,
            Err(error) => {
                on_text(String::new()).map_err(DictationError::Output)?;
                return Err(error.into());
            }
        };
        let mut text = match self.transcribe_samples(audio.samples()) {
            Ok(text) => text,
            Err(error) => {
                on_text(String::new()).map_err(DictationError::Output)?;
                return Err(error.into());
            }
        };
        if text.trim().is_empty() {
            text.clear();
        }
        on_text(text).map_err(DictationError::Output)
    }
}
