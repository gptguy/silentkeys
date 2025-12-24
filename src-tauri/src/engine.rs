use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::asr::{
    current_download_progress, default_model_root, record_failure, resolve_model_dir, AsrError,
    AsrModel, DownloadProgress, InferenceConfig,
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
        if let Ok(mut status) = self.status.lock() {
            if matches!(*status, EngineState::Failed(_)) {
                *status = EngineState::Unloaded;
            }
        }
        self.ensure_model_loaded()
    }

    pub fn ensure_model_loaded(&self) -> Result<(), String> {
        self.update_last_use();

        loop {
            let mut status = self
                .status
                .lock()
                .map_err(|_| "Status lock poisoned".to_string())?;
            match *status {
                EngineState::Loaded => return Ok(()),
                EngineState::Unloaded => {
                    if matches!(*status, EngineState::Loading) {
                        return Ok(());
                    }
                    *status = EngineState::Loading;
                    drop(status);

                    let app_handle = self.app_handle.clone();
                    let model_arc = self.model.clone();
                    let state_arc = self.status.clone();
                    let condvar = self.status_cv.clone();

                    let app_emitter = app_handle.clone().ok_or("No app handle")?;
                    let _ = app_emitter.emit("model_status", "loading");

                    std::thread::spawn(move || {
                        match Self::init_model(&app_handle.unwrap()) {
                            Ok(model) => {
                                if let Ok(mut m) = model_arc.write() {
                                    *m = Some(model);
                                }
                                if let Ok(mut s) = state_arc.lock() {
                                    *s = EngineState::Loaded;
                                }
                                let _ = app_emitter.emit("model_status", "loaded");
                            }
                            Err(e) => {
                                let msg = e.user_message().to_string();
                                log::error!("Load failed: {msg}");
                                record_failure(msg.clone());
                                if let Ok(mut s) = state_arc.lock() {
                                    *s = EngineState::Failed(msg);
                                }
                            }
                        }
                        condvar.notify_all();
                    });
                }
                EngineState::Loading => {
                    let (_s, res) = self
                        .status_cv
                        .wait_timeout(status, Duration::from_secs(30))
                        .map_err(|_| "Wait timeout poisoned")?;
                    if res.timed_out() {
                        return Err("Model load failed: timeout".to_string());
                    }
                }
                EngineState::Failed(_) => {
                    *status = EngineState::Unloaded;
                }
            }
        }
    }

    fn spawn_idle_watcher(&self) {
        let (last_use, model, status) = (
            self.last_use.clone(),
            self.model.clone(),
            self.status.clone(),
        );
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            if let Ok(last) = last_use.lock() {
                if last.elapsed().as_secs() > 300 {
                    if let Ok(mut status_guard) = status.lock() {
                        if matches!(*status_guard, EngineState::Loaded) {
                            log::info!("Unloading idle ASR model");
                            if let Ok(mut m) = model.write() {
                                *m = None;
                            }
                            *status_guard = EngineState::Unloaded;
                        }
                    }
                }
            }
        });
    }

    pub fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        reuse_workspace: bool,
    ) -> Result<String, String> {
        self.ensure_model_loaded()?;
        self.update_last_use();

        let mut model_guard = self.model.write().map_err(|e| e.to_string())?;
        let model = model_guard.as_mut().ok_or("Model not ready")?;

        let config = InferenceConfig::default();
        let transcript = model
            .transcribe_samples(samples, reuse_workspace, Some(config))
            .map_err(|e| e.to_string())?;

        if !transcript.text.trim().is_empty() {
            let char_count = transcript.text.chars().count();
            log::info!("Transcription complete ({} chars)", char_count);
            log::debug!("Transcript text: '{}'", transcript.text);
        }

        Ok(transcript.text)
    }

    pub fn start_streaming(
        &self,
    ) -> Result<std::sync::mpsc::Sender<crate::audio_processing::AudioFrame>, String> {
        use crate::recording::Recorder;
        let app = self.app_handle.as_ref().ok_or("No app handle")?;
        let (tx, rx) = std::sync::mpsc::channel();
        let app_handle = app.clone();
        let model = self.model.clone();

        let pipeline = Arc::new(crate::streaming::StreamingPipeline::new());
        pipeline.start(rx, model, move |patch| {
            if !patch.text.is_empty() {
                Recorder::global().mark_streamed_any();
                let _ = app_handle.emit("transcription_update", patch);
            }
        });
        Ok(tx)
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

        log::info!("Loading ASR from {}", model_dir.display());
        let model = AsrModel::new(&model_dir, true)?;
        log::info!("ASR init took {:?}", start.elapsed());
        Ok(model)
    }
}
