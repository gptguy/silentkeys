use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::asr::{
    default_model_root, invalidate_model_verification, resolve_model_dir_with_progress, AsrError,
    AsrModel,
};
use crate::errors::UserFacing;
use crate::recording::Recorder;
use crate::streaming::{StreamingError, StreamingPipeline, UpdateSink};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error(transparent)]
    Asr(#[from] AsrError),
    #[error("speech engine state is unavailable")]
    StateUnavailable,
    #[error("speech model is unavailable")]
    ModelUnavailable,
    #[error("speech model load timed out")]
    LoadTimeout,
}

impl UserFacing for EngineError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::Asr(error) => error.user_message(),
            Self::LoadTimeout => "The speech model took too long to load. Please try again.",
            Self::StateUnavailable | Self::ModelUnavailable => {
                "The speech engine is unavailable. Please restart the app."
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", content = "message", rename_all = "snake_case")]
pub enum EngineState {
    Unloaded,
    Loading,
    Loaded,
    Failed(String),
}

#[derive(Clone)]
pub struct SpeechEngine {
    model: Arc<RwLock<Option<AsrModel>>>,
    status: Arc<Mutex<EngineState>>,
    status_cv: Arc<Condvar>,
    streaming_pipeline: Arc<StreamingPipeline>,
    app_handle: AppHandle,
    recorder: &'static Recorder,
}

impl SpeechEngine {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            model: Arc::new(RwLock::new(None)),
            status: Arc::new(Mutex::new(EngineState::Unloaded)),
            status_cv: Arc::new(Condvar::new()),
            streaming_pipeline: Arc::new(StreamingPipeline::new()),
            app_handle,
            recorder: Recorder::global(),
        }
    }

    pub(crate) fn recorder(&self) -> &'static Recorder {
        self.recorder
    }

    pub(crate) fn app(&self) -> &AppHandle {
        &self.app_handle
    }

    pub fn is_dictating(&self) -> bool {
        self.recorder.is_recording()
    }

    pub fn state(&self) -> EngineState {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|_| EngineState::Failed("Speech engine state is unavailable.".into()))
    }

    pub fn is_ready(&self) -> bool {
        self.state() == EngineState::Loaded
    }

    pub fn retry_model_download(&self) -> Result<(), EngineError> {
        if let Ok(mut status) = self.status.lock() {
            if matches!(*status, EngineState::Failed(_)) {
                *status = EngineState::Unloaded;
            }
        }
        self.ensure_model_loaded()
    }

    pub fn ensure_model_loaded(&self) -> Result<(), EngineError> {
        loop {
            let mut status = self
                .status
                .lock()
                .map_err(|_| EngineError::StateUnavailable)?;
            match *status {
                EngineState::Loaded => return Ok(()),
                EngineState::Unloaded => {
                    *status = EngineState::Loading;
                    drop(status);
                    self.emit_engine_state(&EngineState::Loading);

                    let app_handle = self.app_handle.clone();
                    let model_arc = self.model.clone();
                    let state_arc = self.status.clone();
                    let condvar = self.status_cv.clone();

                    std::thread::spawn(move || {
                        let outcome = match Self::init_model(&app_handle) {
                            Ok(model) => match model_arc.write() {
                                Ok(mut model_slot) => {
                                    *model_slot = Some(model);
                                    EngineState::Loaded
                                }
                                Err(_) => {
                                    EngineState::Failed("Speech model state is unavailable.".into())
                                }
                            },
                            Err(error) => {
                                log::error!("Speech model init failed: {error}");
                                EngineState::Failed(error.user_message().to_string())
                            }
                        };
                        if let Ok(mut state) = state_arc.lock() {
                            *state = outcome.clone();
                        }
                        if let Err(error) = app_handle.emit("engine_state", outcome) {
                            log::warn!("Could not emit speech engine state: {error}");
                        }
                        condvar.notify_all();
                    });
                }
                EngineState::Loading => {
                    let (_status, wait_result) = self
                        .status_cv
                        .wait_timeout(status, MODEL_LOAD_TIMEOUT)
                        .map_err(|_| EngineError::StateUnavailable)?;
                    if wait_result.timed_out() {
                        return Err(EngineError::LoadTimeout);
                    }
                }
                EngineState::Failed(_) => {
                    *status = EngineState::Unloaded;
                }
            }
        }
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, EngineError> {
        self.ensure_model_loaded()?;

        let mut model_guard = self
            .model
            .write()
            .map_err(|_| EngineError::ModelUnavailable)?;
        let model = model_guard.as_mut().ok_or(EngineError::ModelUnavailable)?;

        let text = model.transcribe_samples(samples)?;

        if !text.trim().is_empty() {
            let char_count = text.chars().count();
            log::info!("Transcription complete ({} chars)", char_count);
        }

        Ok(text)
    }

    pub fn languages(&self) -> Result<Vec<String>, EngineError> {
        let model = self
            .model
            .read()
            .map_err(|_| EngineError::ModelUnavailable)?;
        Ok(model
            .as_ref()
            .ok_or(EngineError::ModelUnavailable)?
            .languages()
            .to_vec())
    }

    pub fn validate_language(&self, language: &str) -> Result<(), EngineError> {
        let model = self
            .model
            .read()
            .map_err(|_| EngineError::ModelUnavailable)?;
        if model
            .as_ref()
            .ok_or(EngineError::ModelUnavailable)?
            .supports_language(language)
        {
            return Ok(());
        }
        Err(AsrError::UnsupportedLanguage(language.to_string()).into())
    }

    pub fn set_language(&self, language: &str) -> Result<String, EngineError> {
        let mut model = self
            .model
            .write()
            .map_err(|_| EngineError::ModelUnavailable)?;
        let selected = model
            .as_mut()
            .ok_or(EngineError::ModelUnavailable)?
            .set_language(language)?;
        log::info!("Nemotron ASR language hint changed to {selected}");
        Ok(selected)
    }

    pub fn start_streaming(
        &self,
        on_update: impl UpdateSink,
    ) -> Result<std::sync::mpsc::Sender<crate::audio_processing::AudioFrame>, StreamingError> {
        if !self.is_ready() {
            return Err(StreamingError::ModelNotReady);
        }

        let (tx, rx) = std::sync::mpsc::channel();
        self.streaming_pipeline
            .start(rx, self.model.clone(), on_update)?;
        Ok(tx)
    }

    pub fn finish_streaming(&self) -> Result<(), StreamingError> {
        self.streaming_pipeline.finish()
    }

    pub fn reset_model_state(&self) {
        if let Ok(mut guard) = self.model.write() {
            if let Some(model) = guard.as_mut() {
                model.reset_state();
                log::info!("ASR model state reset");
            }
        }
    }

    fn init_model(app_handle: &AppHandle) -> Result<AsrModel, AsrError> {
        let start = Instant::now();
        let model_root = default_model_root(app_handle);
        let model_dir = resolve_model_dir_with_progress(&model_root, |progress| {
            if let Err(error) = app_handle.emit("model_download_progress", progress) {
                log::warn!("Could not emit model download progress: {error}");
            }
        })?;
        let language = crate::settings::get_settings(app_handle).asr_language;

        log::info!("Loading ASR from {}", model_dir.display());
        let model = AsrModel::new(&model_dir, &language).inspect_err(|_| {
            invalidate_model_verification(&model_dir);
        })?;
        if !model.supports_language(&language) {
            let mut settings = crate::settings::get_settings(app_handle);
            settings.asr_language = crate::settings::DEFAULT_ASR_LANGUAGE.to_string();
            if let Err(error) = crate::settings::save_settings(app_handle, &settings) {
                log::warn!("Could not repair unsupported speech language setting: {error}");
            }
        }
        log::info!("ASR init took {:?}", start.elapsed());
        Ok(model)
    }

    fn emit_engine_state(&self, state: &EngineState) {
        if let Err(error) = self.app_handle.emit("engine_state", state) {
            log::warn!("Could not emit speech engine state: {error}");
        }
    }
}
