use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::asr::{
    current_download_progress, default_model_root, record_failure, resolve_model_dir, AsrError,
    AsrModel, DownloadProgress, InferenceConfig, Transcript,
};
use tauri::{AppHandle, Emitter};

#[derive(Clone, Debug, PartialEq)]
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
    last_use: Arc<Mutex<Instant>>,
    app_handle: Option<AppHandle>,
}

impl SpeechEngine {
    pub fn new(app_handle: AppHandle) -> Self {
        let engine = Self {
            model: Arc::new(RwLock::new(None)),
            status: Arc::new(Mutex::new(EngineState::Unloaded)),
            status_cv: Arc::new(Condvar::new()),
            last_use: Arc::new(Mutex::new(Instant::now())),
            app_handle: Some(app_handle),
        };
        engine.spawn_idle_watcher();
        engine
    }

    pub fn get_model(&self) -> Arc<RwLock<Option<AsrModel>>> {
        self.model.clone()
    }

    pub fn is_ready(&self) -> bool {
        self.status
            .lock()
            .map(|s| matches!(*s, EngineState::Loaded))
            .unwrap_or(false)
    }

    fn update_last_use(&self) {
        if let Ok(mut last) = self.last_use.lock() {
            *last = Instant::now();
        }
    }

    pub fn download_progress(&self) -> Option<DownloadProgress> {
        current_download_progress()
    }

    pub fn retry_model_download(&self) -> Result<(), String> {
        self.ensure_model_loaded()
    }

    pub fn ensure_model_loaded(&self) -> Result<(), String> {
        self.update_last_use();

        let mut status = self
            .status
            .lock()
            .map_err(|_| "Status lock poisoned".to_string())?;
        match *status {
            EngineState::Loaded => return Ok(()),
            EngineState::Failed(ref e) => {
                log::warn!("Previous load failed: {}. Retrying...", e);
                *status = EngineState::Unloaded;
            }
            EngineState::Loading => {
                log::debug!("Waiting for model load...");
                let result = self
                    .status_cv
                    .wait_timeout(status, Duration::from_secs(30))
                    .map_err(|_| "Status lock poisoned during wait".to_string())?;
                status = result.0;
                if result.1.timed_out() {
                    return Err("Model load timed out".to_string());
                }
                match *status {
                    EngineState::Loaded => return Ok(()),
                    EngineState::Failed(ref e) => return Err(e.clone()),
                    _ => return Err("Model load state invalid after wait".to_string()),
                }
            }
            EngineState::Unloaded => {}
        }

        *status = EngineState::Loading;
        drop(status);

        let app_handle = self.app_handle.clone().ok_or("No app handle")?;
        let model_safe = self.model.clone();
        let status_safe = self.status.clone();
        let cv_safe = self.status_cv.clone();

        let _ = app_handle.emit("model_status", "loading");

        std::thread::spawn(move || match Self::init_model(&app_handle) {
            Ok(loaded_model) => {
                let Ok(mut model_guard) = model_safe.write() else {
                    log::error!("Model lock poisoned during load");
                    return;
                };
                *model_guard = Some(loaded_model);
                drop(model_guard);

                let Ok(mut s) = status_safe.lock() else {
                    log::error!("Status lock poisoned after load");
                    return;
                };
                *s = EngineState::Loaded;
                log::info!("Model loaded successfully");
                let _ = app_handle.emit("model_status", "loaded");
                cv_safe.notify_all();
            }
            Err(e) => {
                let msg = e.user_message().to_string();
                log::error!("Load failed: {msg} ({e})");
                record_failure(msg.clone());
                if let Ok(mut s) = status_safe.lock() {
                    *s = EngineState::Failed(msg);
                }
                cv_safe.notify_all();
            }
        });

        let mut status = self
            .status
            .lock()
            .map_err(|_| "Status lock poisoned".to_string())?;
        loop {
            match *status {
                EngineState::Loaded => return Ok(()),
                EngineState::Failed(ref e) => return Err(e.clone()),
                EngineState::Loading => {
                    let result = self
                        .status_cv
                        .wait_timeout(status, Duration::from_secs(30))
                        .map_err(|_| "Status lock poisoned during wait".to_string())?;
                    status = result.0;
                    if result.1.timed_out() {
                        return Err("Model load timed out".to_string());
                    }
                }
                _ => return Err("Unexpected state during load".to_string()),
            }
        }
    }

    fn spawn_idle_watcher(&self) {
        let last_use = self.last_use.clone();
        let model = self.model.clone();
        let status = self.status.clone();

        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            let elapsed = match last_use.lock() {
                Ok(guard) => guard.elapsed(),
                Err(_) => continue,
            };
            if elapsed.as_secs() > 300 {
                let Ok(mut status_guard) = status.lock() else {
                    continue;
                };
                if matches!(*status_guard, EngineState::Loaded) {
                    log::info!("Unloading idle ASR model after {:?}", elapsed);
                    if let Ok(mut m) = model.write() {
                        *m = None;
                    }
                    *status_guard = EngineState::Unloaded;
                }
            }
        });
    }

    pub fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        reuse_workspace: bool,
    ) -> Result<String, String> {
        let result =
            self.transcribe_samples_inner(&samples, reuse_workspace, InferenceConfig::from_env())?;

        if !result.text.trim().is_empty() {
            let char_count = result.text.chars().count();
            log::info!("Transcription complete ({} chars)", char_count);
            log::debug!("Transcript text: '{}'", result.text);
        }

        Ok(result.text)
    }

    pub fn transcribe_samples_with_config(
        &self,
        samples: &[f32],
        reuse_workspace: bool,
        config: &InferenceConfig,
    ) -> Result<Transcript, String> {
        self.transcribe_samples_inner(samples, reuse_workspace, config.clone())
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
        let model_dir = resolve_model_dir(&model_root)?;

        if let Err(e) = crate::asr::ensure_vad_model(app_handle) {
            log::warn!("VAD check failed: {}", e);
        }

        log::info!("Loading ASR from {}", model_dir.display());
        let model = AsrModel::new(&model_dir, true)?;
        log::info!("ASR init took {:?}", start.elapsed());
        Ok(model)
    }

    fn transcribe_samples_inner(
        &self,
        samples: &[f32],
        reuse_workspace: bool,
        config: InferenceConfig,
    ) -> Result<Transcript, String> {
        if samples.is_empty() {
            return Err("No audio captured.".to_string());
        }

        log::debug!(
            "transcribe_samples: samples={}, reuse_workspace={}",
            samples.len(),
            reuse_workspace
        );

        self.ensure_model_loaded()?;

        let mut model_guard = self
            .model
            .write()
            .map_err(|_| "Model lock poisoned".to_string())?;
        let model: &mut AsrModel = model_guard
            .as_mut()
            .ok_or_else(|| "Speech model is not loaded yet. Please try again.".to_string())?;

        self.update_last_use();

        let start = Instant::now();
        let result = model
            .transcribe_samples_ref(samples, reuse_workspace, Some(config))
            .map_err(|err| {
                let msg = err.user_message().to_string();
                log::error!("Transcription error: {}", msg);
                log::debug!("Transcription error detail: {err}");
                msg
            })?;

        log::debug!("Transcription took {:?}", start.elapsed());
        Ok(result)
    }
}
