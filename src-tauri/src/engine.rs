//! High-level speech recognition engine facade.

use std::sync::Mutex;
use std::time::Instant;

use crate::asr::{
    current_download_progress, default_model_root, record_failure, resolve_model_dir, AsrError,
    AsrModel, DownloadProgress,
};
use tauri::AppHandle;

#[derive(Default)]
pub struct SpeechEngine {
    model: Mutex<Option<AsrModel>>,
    app_handle: Option<AppHandle>,
}

impl SpeechEngine {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            model: Mutex::new(None),
            app_handle: Some(app_handle),
        }
    }
    pub fn is_ready(&self) -> bool {
        self.model
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    pub fn download_progress(&self) -> Option<DownloadProgress> {
        current_download_progress()
    }

    pub fn retry_model_download(&self) -> Result<(), String> {
        self.ensure_model_loaded()
    }

    pub fn ensure_model_loaded(&self) -> Result<(), String> {
        if self.is_ready() {
            return Ok(());
        }

        let app_handle = self
            .app_handle
            .as_ref()
            .ok_or_else(|| "SpeechEngine not initialized with AppHandle".to_string())?;

        let model = Self::init_model(app_handle).map_err(|err| {
            let message = err.user_message().to_string();
            record_failure(message.clone());
            log::error!("Failed to initialize ASR model: {err}");
            message
        })?;

        let mut guard = self
            .model
            .lock()
            .map_err(|_| "The speech engine is busy. Please try again.".to_string())?;

        if guard.is_none() {
            *guard = Some(model);
        }

        Ok(())
    }

    pub fn transcribe_samples(&self, samples: Vec<f32>) -> Result<String, String> {
        if samples.is_empty() {
            return Err("No audio captured.".to_string());
        }

        self.ensure_model_loaded()?;

        let mut guard = self
            .model
            .lock()
            .map_err(|_| "The speech engine is busy. Please try again.".to_string())?;

        let model = guard
            .as_mut()
            .ok_or_else(|| "Speech model is not loaded yet. Please try again.".to_string())?;

        let start = Instant::now();
        let result = model.transcribe_samples(samples).map_err(|err| {
            let message = err.user_message().to_string();
            log::error!("Transcription failed: {}", err);
            message
        })?;

        log::info!("Transcription completed in {:?}", start.elapsed());
        Ok(result.text)
    }

    fn init_model(app_handle: &AppHandle) -> Result<AsrModel, AsrError> {
        let start = Instant::now();
        let model_root = default_model_root(app_handle);
        log::info!("Resolving ASR model under {}", model_root.display());

        let model_dir = resolve_model_dir(&model_root)?;
        log::info!(
            "Resolved ASR snapshot to {} (in {:?})",
            model_dir.display(),
            start.elapsed()
        );

        let load_start = Instant::now();
        let model = AsrModel::new(&model_dir, true)?;
        log::info!(
            "ASR model loaded from {} in {:?}",
            model_dir.display(),
            load_start.elapsed()
        );

        Ok(model)
    }
}
